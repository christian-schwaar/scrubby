// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::Command;
use std::path::Path;
 

#[tauri::command]
async fn run_engine(
    input: String,
    mode: String,
    output_dir: String,
    language: String,
    ocr: bool,
    filters: Option<serde_json::Value>,
    threshold: Option<f64>,
) -> Result<String, String> {
    // Setze OCR Environment Variable
    if ocr {
        std::env::set_var("OCR", "1");
    } else {
        std::env::set_var("OCR", "0");
    }

    // Führe die neue Python-Engine (engineV2.py) aus
    // bevorzuge lokales venv-Python (relativ zum Projekt-Root), sonst fallback auf system python3
    // Achtung: Wir setzen unten current_dir(".."), daher müssen die Pfade RELATIV zum Projekt-Root sein (ohne ../ Präfix).
    let project_root = Path::new("..");
    // Bevorzuge venv311 (Python 3.11) vor venv (Python 3.14), da einige Wheels stabiler sind
    let candidates: [(&str, std::path::PathBuf); 4] = [
        ("venv311/bin/python3", project_root.join("venv311").join("bin").join("python3")),
        ("venv311/bin/python", project_root.join("venv311").join("bin").join("python")),
        ("venv/bin/python3", project_root.join("venv").join("bin").join("python3")),
        ("venv/bin/python", project_root.join("venv").join("bin").join("python")),
    ];
    let mut python_cmd: String = String::from("python3");
    for (rel_cmd, check_path) in candidates.iter() {
        if check_path.exists() {
            python_cmd = rel_cmd.to_string();
            break;
        }
    }
    let mut cmd = Command::new(&python_cmd);
    // Übergib Filter als ENV-Variable (JSON)
    if let Some(f) = filters {
        cmd.env("FILTERS", f.to_string());
    }
    let output = cmd
        // Run the engine from the project root so it writes to ../data and not src-tauri/data
        .current_dir("..")
        .arg("engine/engineV2.py")
        .arg("--input")
        .arg(&input)
        .arg("--mode")
        .arg(&mode)
        .arg("--outdir")
        .arg(&output_dir)
        .arg("--language")
        .arg(&language)
        .arg("--threshold")
        .arg(threshold.unwrap_or(0.6).to_string())
        .arg("--verbose")
        .output()
        .map_err(|e| format!("Failed to execute engine: {}", e))?;

    if !output.status.success() {
        let stderr_msg = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout_msg = String::from_utf8_lossy(&output.stdout).trim().to_string();

        // Try to extract a meaningful error message
        let message = if !stderr_msg.is_empty() {
            stderr_msg
        } else if stdout_msg.starts_with('{') {
            // Try to parse JSON { ok: false, error: "..." }
            match serde_json::from_str::<serde_json::Value>(&stdout_msg) {
                Ok(v) => v
                    .get("error")
                    .and_then(|e| e.as_str())
                    .unwrap_or("Unknown error")
                    .to_string(),
                Err(_) => stdout_msg,
            }
        } else if !stdout_msg.is_empty() {
            stdout_msg
        } else {
            "Unknown error".to_string()
        };

        return Err(format!("Engine execution failed: {}", message));
    }

    let result = String::from_utf8_lossy(&output.stdout);
    Ok(result.to_string())
}

#[tauri::command]
async fn open_file_dialog(app: tauri::AppHandle) -> Result<Vec<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = std::sync::mpsc::channel();

    app.dialog()
        .file()
        .add_filter("Supported Files", &["pdf", "png", "jpg", "jpeg", "txt", "md", "json"])
        .pick_file(move |result| {
            let _ = tx.send(result);
        });

    let file_path = rx
        .recv()
        .map_err(|e| format!("Dialog error: {}", e))?
        .ok_or("No file selected")?;

    Ok(vec![file_path.to_string()])
}

#[tauri::command]
async fn open_folder(path: String) -> Result<(), String> {
    let path = Path::new(&path);
    
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }
    
    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }
    
    #[cfg(target_os = "windows")]
    {
        Command::new("explorer")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }
    
    Ok(())
}

#[tauri::command]
async fn open_file(path: String) -> Result<(), String> {
    let path = Path::new(&path);
    
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("Failed to open file: {}", e))?;
    }
    
    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("Failed to open file: {}", e))?;
    }
    
    #[cfg(target_os = "windows")]
    {
        Command::new("explorer")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("Failed to open file: {}", e))?;
    }
    
    Ok(())
}

#[tauri::command]
async fn write_temp_file(file_name: String, data_base64: String) -> Result<String, String> {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    // Decode base64
    let bytes = base64::decode(&data_base64).map_err(|e| format!("Base64 decode error: {}", e))?;

    // Build target directory in project root (one level up from src-tauri): ../data/input/<epoch>
    let epoch = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e| e.to_string())?.as_secs();
    let base_dir = Path::new("..").join("data").join("input");
    let target_dir = base_dir.join(format!("{}", epoch));
    fs::create_dir_all(&target_dir).map_err(|e| format!("Create dir error: {}", e))?;

    let target_path = target_dir.join(file_name);
    fs::write(&target_path, bytes).map_err(|e| format!("Write file error: {}", e))?;

    // Return absolute path to be safe regardless of child current_dir
    let abs = std::fs::canonicalize(&target_path).map_err(|e| format!("Canonicalize error: {}", e))?;
    Ok(abs.to_string_lossy().to_string())
}

#[tauri::command]
async fn apply_redactions(input: String, redactions_json: String) -> Result<String, String> {
    use std::path::PathBuf;
    // Reuse python discovery
    let project_root = std::path::Path::new("..");
    let candidates: [(&str, std::path::PathBuf); 4] = [
        ("venv311/bin/python3", project_root.join("venv311").join("bin").join("python3")),
        ("venv311/bin/python", project_root.join("venv311").join("bin").join("python")),
        ("venv/bin/python3", project_root.join("venv").join("bin").join("python3")),
        ("venv/bin/python", project_root.join("venv").join("bin").join("python")),
    ];
    let mut python_cmd: String = String::from("python3");
    for (rel_cmd, check_path) in candidates.iter() {
        if check_path.exists() {
            python_cmd = rel_cmd.to_string();
            break;
        }
    }
    let mut cmd = std::process::Command::new(&python_cmd);
    let output = cmd
        .current_dir("..")
        .arg("engine/redact.py")
        .arg("--input")
        .arg(&input)
        .arg("--redactions")
        .arg(&redactions_json)
        .output()
        .map_err(|e| format!("Failed to execute redact.py: {}", e))?;

    if !output.status.success() {
        let stderr_msg = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout_msg = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let message = if !stderr_msg.is_empty() { stderr_msg } else { stdout_msg };
        return Err(format!("Redaction failed: {}", message));
    }
    let result = String::from_utf8_lossy(&output.stdout);
    Ok(result.to_string())
}
#[tauri::command]
async fn get_file_size(path: String) -> Result<u64, String> {
    let meta = std::fs::metadata(&path).map_err(|e| format!("Metadata error: {}", e))?;
    Ok(meta.len())
}

#[tauri::command]
async fn read_file_base64(path: String) -> Result<String, String> {
    use std::fs;
    let bytes = fs::read(&path).map_err(|e| format!("Read error: {}", e))?;
    Ok(base64::encode(bytes))
}
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![
            run_engine,
            open_file_dialog,
            open_folder,
            open_file,
            get_file_size,
            write_temp_file,
            apply_redactions,
            read_file_base64
        ])
        .run(tauri::generate_context!())
        .expect("error running tauri application");
}
