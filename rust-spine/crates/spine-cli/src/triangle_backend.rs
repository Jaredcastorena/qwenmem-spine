//! Bounded, fail-closed owner of the local triangle inference process.
use serde_json::{Value, json};
use std::{ffi::OsString, io, path::Path, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
};

pub const MAX_LINE: usize = 2 * 1024 * 1024;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub struct Backend {
    child: Option<Child>,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    timeout: Duration,
    next_id: u64,
}

impl Backend {
    pub fn spawn(program: &Path, args: &[OsString], timeout: Duration) -> Result<Self> {
        let mut child = Command::new(program)
            .args(args)
            // Decryption belongs to Rust; the inference child receives evidence only.
            .env_remove("SPINE_HEART_PASSPHRASE")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()?;
        let input = child.stdin.take().ok_or("backend stdin unavailable")?;
        let output = BufReader::new(child.stdout.take().ok_or("backend stdout unavailable")?);
        Ok(Self {
            child: Some(child),
            input,
            output,
            timeout,
            next_id: 1,
        })
    }

    pub async fn call(&mut self, op: &str, args: Value) -> Result<Value> {
        if self.child.is_none() {
            return Err("triangle backend is stopped".into());
        }
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or("request id exhausted")?;
        let mut wire = serde_json::to_vec(&json!({"schema":1,"id":id,"op":op,"args":args}))?;
        if wire.len() >= MAX_LINE {
            return Err("triangle request exceeds 2 MiB wire budget".into());
        }
        wire.push(b'\n');
        let exchange = async {
            self.input.write_all(&wire).await?;
            self.input.flush().await?;
            let bytes = read_line_bounded(&mut self.output).await?;
            validate_response(&bytes, id)
        };
        let result = tokio::select! {
            response = tokio::time::timeout(self.timeout, exchange) => match response {
                Ok(result) => result,
                Err(_) => Err("triangle backend timed out; no interaction committed".into()),
            },
            signal = tokio::signal::ctrl_c() => {
                signal?;
                Err("triangle backend cancelled; no interaction committed".into())
            }
        };
        if result.is_err() {
            self.abort().await;
        }
        result
    }

    pub async fn abort(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        if self.child.is_none() {
            return Ok(());
        }
        let result = self.call("shutdown", json!({})).await;
        if result.is_ok()
            && let Some(child) = self.child.as_mut()
        {
            match tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
                Ok(Ok(status)) if status.success() => {
                    self.child.take();
                    return Ok(());
                }
                _ => {
                    self.abort().await;
                    return Err("triangle backend did not shut down cleanly".into());
                }
            }
        }
        self.abort().await;
        result.map(|_| ())
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.start_kill();
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                runtime.spawn(async move {
                    let _ = child.wait().await;
                });
            }
            // kill_on_drop remains set; Tokio's process reaper owns abandoned waits.
        }
    }
}

async fn read_line_bounded<R: tokio::io::AsyncBufRead + Unpin>(reader: &mut R) -> Result<Vec<u8>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Err("triangle backend closed stdout before a complete response".into());
        }
        let count = available
            .iter()
            .position(|b| *b == b'\n')
            .map_or(available.len(), |i| i + 1);
        if line.len() + count > MAX_LINE {
            return Err("triangle response exceeds 2 MiB wire budget".into());
        }
        let done = available[count - 1] == b'\n';
        line.extend_from_slice(&available[..count]);
        reader.consume(count);
        if done {
            return Ok(line);
        }
    }
}

fn validate_response(bytes: &[u8], id: u64) -> Result<Value> {
    let response: Value = serde_json::from_slice(bytes)?;
    if response.get("schema").and_then(Value::as_u64) != Some(1)
        || response.get("id").and_then(Value::as_u64) != Some(id)
    {
        return Err("triangle response schema/request id mismatch".into());
    }
    match response.get("ok").and_then(Value::as_bool) {
        Some(true) => response
            .get("result")
            .filter(|v| v.is_object())
            .cloned()
            .ok_or_else(|| "triangle response is missing its result object".into()),
        Some(false) => {
            let error = &response["error"];
            let code = error["code"].as_str().ok_or("triangle error has no code")?;
            let message = error["message"]
                .as_str()
                .ok_or("triangle error has no message")?;
            Err(io::Error::other(format!("triangle backend {code}: {message}")).into())
        }
        None => Err("triangle response has no boolean ok field".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn triangle_response_requires_identity_and_complete_object() {
        assert!(validate_response(br#"{"schema":1,"id":2,"ok":true,"result":{}}"#, 1).is_err());
        assert!(validate_response(br#"{"schema":1,"id":1,"ok":true,"result":[]}"#, 1).is_err());
        assert!(
            validate_response(
                br#"{"schema":1,"id":1,"ok":false,"error":{"code":"bad","message":"bad"}}"#,
                1
            )
            .is_err()
        );
        assert!(validate_response(br#"{"schema":1,"id":1,"ok":true,"result":{}}"#, 1).is_ok());
    }
    #[tokio::test]
    async fn triangle_response_requires_newline_and_enforces_size() {
        assert!(read_line_bounded(&mut &b"{}"[..]).await.is_err());
        let bytes = vec![b'x'; MAX_LINE + 1];
        assert!(read_line_bounded(&mut bytes.as_slice()).await.is_err());
        assert_eq!(read_line_bounded(&mut &b"{}\n"[..]).await.unwrap(), b"{}\n");
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn triangle_timeout_kills_and_reaps_child() {
        // Tests alone use a fake executable; production has no inference fallback.
        let mut backend = Backend::spawn(
            Path::new("/bin/sleep"),
            &["30".into()],
            Duration::from_millis(20),
        )
        .unwrap();
        assert!(backend.call("hello", json!({})).await.is_err());
        assert!(backend.child.is_none());
    }
}
