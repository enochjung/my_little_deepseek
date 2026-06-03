mod kv_cache;
mod special_token;

use crate::Model;
pub(crate) use kv_cache::KVCache;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;

pub struct Session<'a> {
    model: &'a Model,
    tokens: Vec<u32>,
    kv_cache: KVCache,
}

impl<'a> Session<'a> {
    /// Creates a new task by appending a user prompt to this session.
    ///
    /// The returned task starts its worker immediately.
    pub fn send_prompt(self, user_input: &str) -> Result<SessionTask<'a>, crate::Error> {
        // Build the prompt tokens that will be prefed into the worker.
        let model = self.model;
        let mut tokens = self.tokens;
        let kv_cache = self.kv_cache;

        tokens.push(special_token::USER);
        tokens.append(&mut self.model.tokenize(user_input)?);
        tokens.push(special_token::ASSISTANT);
        tokens.push(special_token::THINK_START);

        Ok(SessionTask::new(model, tokens, kv_cache))
    }

    pub(crate) fn new(model: &'a Model) -> Self {
        let tokens = vec![special_token::BEGIN_OF_SENTENCE; 1];
        let kv_cache = KVCache::new();

        Self {
            model,
            tokens,
            kv_cache,
        }
    }
}

pub struct SessionTask<'a> {
    model: &'a Model,
    tokens: Vec<u32>,
    worker: Option<JoinHandle<Result<KVCache, crate::Error>>>,
    abort_flag: Arc<AtomicBool>,
    token_rx: mpsc::Receiver<u32>,
}

impl<'a> SessionTask<'a> {
    fn new(model: &'a Model, tokens: Vec<u32>, kv_cache: KVCache) -> Self {
        let prefill_start = kv_cache.len().min(tokens.len());
        let prefill_tokens = tokens[prefill_start..].to_vec();
        let abort_flag = Arc::new(AtomicBool::new(false));
        let (token_tx, token_rx) = mpsc::channel();
        let worker = Worker::spawn(
            model,
            prefill_tokens,
            kv_cache,
            Arc::clone(&abort_flag),
            token_tx,
        );

        Self {
            model,
            tokens,
            worker: Some(worker),
            abort_flag,
            token_rx,
        }
    }

    /// Returns true once the worker thread has stopped.
    pub fn is_finished(&self) -> bool {
        self.worker.as_ref().is_some_and(|w| w.is_finished())
    }

    /// Stops background work and converts this task back into a reusable `Session`.
    pub fn finish_decoding(mut self) -> Result<Session<'a>, crate::Error> {
        self.abort_flag.store(true, Ordering::Relaxed);
        while let Ok(token_id) = self.token_rx.try_recv() {
            self.tokens.push(token_id);
        }
        let tokens = std::mem::take(&mut self.tokens);

        let kv_cache = self
            .worker
            .take()
            .unwrap()
            .join()
            .expect("joining session worker should succeed")?;

        Ok(Session {
            model: self.model,
            tokens,
            kv_cache,
        })
    }

    /// Returns at most one generated token id as a readable string.
    pub fn get_next_string(&mut self) -> Option<String> {
        // TODO: return readable string.
        let token_id = self.token_rx.try_recv().ok()?;
        self.tokens.push(token_id);

        Some(render_token_id(token_id))
    }
}

impl<'a> Drop for SessionTask<'a> {
    fn drop(&mut self) {
        self.abort_flag.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .expect("joining session worker should succeed")
                .expect("decode should succeed");
        }
    }
}

fn render_token_id(token_id: u32) -> String {
    match token_id {
        special_token::BEGIN_OF_SENTENCE => "<begin_of_sentence>".to_string(),
        special_token::USER => "<user>".to_string(),
        special_token::ASSISTANT => "<assistant>".to_string(),
        special_token::THINK_START => "<think>".to_string(),
        special_token::THINK_END => "</think>".to_string(),
        special_token::END_OF_SENTENCE => "<end_of_sentence>".to_string(),
        _ => format!("<{token_id}>"),
    }
}

struct Worker {
    // SAFETY: the worker is joined before the owning SessionTask is dropped.
    model_ptr: usize,
    prefill_tokens: Vec<u32>,
    kv_cache: KVCache,
    abort_flag: Arc<AtomicBool>,
    transmitter: mpsc::Sender<u32>,
}

impl Worker {
    fn spawn(
        model: &Model,
        prefill_tokens: Vec<u32>,
        kv_cache: KVCache,
        abort_flag: Arc<AtomicBool>,
        transmitter: mpsc::Sender<u32>,
    ) -> JoinHandle<Result<KVCache, crate::Error>> {
        let worker = Self {
            model_ptr: model as *const Model as usize,
            prefill_tokens,
            kv_cache,
            abort_flag,
            transmitter,
        };

        std::thread::spawn(move || worker.run())
    }

    fn run(self) -> Result<KVCache, crate::Error> {
        let model = unsafe { &*(self.model_ptr as *const Model) };
        let prefill_tokens = self.prefill_tokens;
        let mut kv_cache = self.kv_cache;
        let abort_flag = self.abort_flag;
        let transmitter = self.transmitter;

        let next_token = model.prefill(&prefill_tokens, &mut kv_cache)?;
        transmitter
            .send(next_token)
            .expect("transmitter send should succeed");
        if next_token == special_token::END_OF_SENTENCE {
            return Ok(kv_cache);
        }

        let mut current_token = next_token;
        loop {
            if abort_flag.load(Ordering::Relaxed) {
                return Ok(kv_cache);
            }

            let next_token = model.decode(current_token, &mut kv_cache)?;
            transmitter
                .send(next_token)
                .expect("transmitter send should succeed");
            if next_token == special_token::END_OF_SENTENCE {
                return Ok(kv_cache);
            }

            current_token = next_token;
        }
    }
}
