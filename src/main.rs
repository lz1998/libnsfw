use std::env::args;
use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::mpsc::{channel, Sender};

use anyhow::Result;
use axum::{Extension, extract, Json};
use axum::body::{Bytes};
use axum::http::StatusCode;
use axum::routing::post;
use image::imageops::{self, FilterType};
use onnxruntime::{
    environment::Environment, GraphOptimizationLevel, ndarray::Array4, session::Session,
    tensor::OrtOwnedTensor,
};
use serde::Serialize;
use tokio::sync::{Mutex, oneshot};

type TaskType = (oneshot::Sender<Result<Prediction>>, Vec<u8>);

#[derive(Debug, Serialize)]
struct Prediction {
    drawings: f32,
    hentai: f32,
    neutral: f32,
    porn: f32,
    sexy: f32,
}

struct Predictor {
    worker_channel: Sender<TaskType>,
}

impl Predictor {
    fn new(model_path: String, worker_thread: u16) -> Self {
        let (tx, rx) = channel::<TaskType>();
        tokio::task::spawn_blocking(move || {
            let environment = Environment::builder().with_name("nsfw").build().unwrap();
            let mut session = environment
                .new_session_builder()
                .unwrap()
                .with_optimization_level(GraphOptimizationLevel::All)
                .unwrap()
                .with_number_threads(worker_thread as i16)
                .unwrap()
                .with_model_from_file(model_path)
                .expect("read model failed");
            for (result_tx, task) in rx.into_iter() {
                result_tx.send(real_predict(&mut session, task)).unwrap();
            }
        });
        Predictor { worker_channel: tx }
    }

    #[allow(dead_code)]
    async fn predict(&self, image_bytes: Vec<u8>) -> Result<Prediction> {
        let (tx, rx) = oneshot::channel();
        self.worker_channel.send((tx, image_bytes)).unwrap();
        rx.await.unwrap()
    }
}

fn real_predict(session: &mut Session, image_bytes: Vec<u8>) -> Result<Prediction> {
    let tensor = Array4::from_shape_vec((1, 224, 224, 3), image_bytes)?.mapv(|x| x as f32 / 255.);
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![tensor])?;
    let output = outputs[0].view();
    let result = output.as_slice().unwrap();

    Ok(Prediction {
        drawings: result[0],
        hentai: result[1],
        neutral: result[2],
        porn: result[3],
        sexy: result[4],
    })
}

const HELPER: &str = r#"LibNSFW
Minimal HTTP server provides nsfw image detection.
more detail in https://github.com/zkonge/libnsfw

Usage:
    ./libnsfw bind_addr nsfw_model worker_thread
Example:
    ./libnsfw 127.0.0.1:8000 ./nsfw.onnx 4

HTTP Request:
    Just POST form-data with `image` key.
HTTP Response:
    {
        "drawings": 0.5251695,
        "hentai": 0.47225672,
        "neutral": 0.0011893457,
        "porn": 0.0011269405,
        "sexy": 0.00025754774
    }
"#;

#[tokio::main]
async fn main() -> Result<()> {
    let args = args();
    if args.len() < 4 {
        println!("{}", HELPER);
        return Ok(());
    }
    let mut args = args.skip(1);
    let bind_addr = args.next().expect("bind_addr not found");
    let model_path = args.next().expect("nsfw_model not found");
    let worker_thread: u16 = args
        .next()
        .expect("worker_thread not found")
        .parse()
        .expect("parse worker_thread failed");

    let predictor = Arc::new(Mutex::new(Predictor::new(model_path, worker_thread)));

    let app = axum::Router::new().route("/", post(upload)).layer(Extension(predictor));
    axum::Server::bind(&SocketAddr::from_str(&bind_addr).expect("error addr"))
        .serve(app.into_make_service())
        .await
        .unwrap();
    Ok(())
}

async fn upload(mut multipart: extract::Multipart, Extension(predictor): Extension<Arc<Mutex<Predictor>>>) -> Result<Json<Prediction>, StatusCode> {
    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap_or_default().to_string();
        let data = match name.as_str() {
            "image" => { field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)? }
            "url" => {
                http_get(&field.text().await.map_err(|_| StatusCode::BAD_REQUEST)?).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            }
            &_ => {
                continue
            }
        };

        let img = image::load_from_memory(&data).map_err(|_|StatusCode::BAD_REQUEST)?;
        let resized_img = imageops::resize(&img.to_rgb8(), 224, 224, FilterType::Lanczos3);
        // let prediction = predictor.lock().await.predict(resized_img.to_vec()).await;
        let prediction = {
            let (tx, rx) = oneshot::channel();
            predictor.lock().await.worker_channel.send((tx, resized_img.to_vec())).unwrap();
            rx.await.unwrap()
        };
        println!("Length of `{}` is {} bytes", name, data.len());

        return prediction.map(|p| Json(p)).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR);
    }
    Err(StatusCode::BAD_REQUEST)
}

async fn http_get(url: &str) -> Result<Bytes> {
    reqwest::get(url)
        .await?
        .bytes()
        .await.map_err(|_| anyhow::anyhow!("failed to get bytes"))
}