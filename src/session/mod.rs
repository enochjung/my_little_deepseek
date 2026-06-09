mod kv_cache;
mod special_token;

use crate::Model;
use crate::device::{Device, DeviceOps, OwnedDevice};
use crate::tensor::ElemType;
pub(crate) use kv_cache::KVCache;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};

#[allow(private_bounds)]
pub struct Session<'a, E: ElemType, ED: Device, TD: Device> {
    model: &'a Model<E, ED, TD>,
    tokens: Vec<u32>,
    kv_caches: Vec<KVCache<E, TD::Base>>,
}

#[allow(private_bounds)]
impl<'a, E: ElemType, ED: Device, TD: Device> Session<'a, E, ED, TD> {
    /// Creates a new task by appending a user prompt to this session.
    ///
    /// The returned task starts automatically.
    pub fn send_prompt<'scope>(
        self,
        scope: &'scope std::thread::Scope<'scope, '_>,
        user_input: &str,
    ) -> Result<SessionTask<'a, E, ED, TD>, crate::Error>
    where
        'a: 'scope,
        ED::Base: DeviceOps<E>,
        TD::Base: DeviceOps<E>,
        ED: Device<Base = TD::Base>,
    {
        let model = self.model;
        let mut tokens = self.tokens;
        let kv_caches = self.kv_caches;

        tokens.push(special_token::USER);
        tokens.append(&mut self.model.tokenize(user_input)?);
        tokens.push(special_token::ASSISTANT);
        let decoded_cursor = tokens.len();
        tokens.push(special_token::THINK_START);
        tokens.push(special_token::NEXT_LINE);

        let prefill_start = (kv_caches[0].n() as usize).min(tokens.len());
        let prefill_tokens = tokens[prefill_start..].to_vec();
        let abort_flag = Arc::new(AtomicBool::new(false));
        let finished_flag = Arc::new(AtomicBool::new(false));
        let (token_tx, token_rx) = mpsc::channel();
        let (kv_tx, kv_rx) = mpsc::channel();

        let abort_clone = Arc::clone(&abort_flag);
        let finished_clone = Arc::clone(&finished_flag);

        scope.spawn(move || {
            let result = run(model, prefill_tokens, kv_caches, abort_clone, token_tx);
            finished_clone.store(true, Ordering::Release);
            let _ = kv_tx.send(result);
        });

        Ok(SessionTask {
            model,
            tokens,
            decoded_cursor,
            abort_flag,
            finished_flag,
            token_rx,
            kv_rx,
        })
    }

    pub(crate) fn new(model: &'a Model<E, ED, TD>) -> Result<Self, crate::Error> {
        const INITIAL_N: usize = 1024;

        let tokens = vec![special_token::BEGIN_OF_SENTENCE; 1];
        let kv_caches = (0..model.num_hidden_layers)
            .map(|_| {
                let k_device = TD::Base::new(
                    INITIAL_N * model.head_size as usize * model.num_key_value_heads * E::BYTES,
                )?;
                let v_device = TD::Base::new(
                    INITIAL_N * model.head_size as usize * model.num_key_value_heads * E::BYTES,
                )?;

                KVCache::<E, _>::new(
                    k_device,
                    v_device,
                    model.head_size,
                    model.num_key_value_heads,
                    0,
                )
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            model,
            tokens,
            kv_caches,
        })
    }
}

#[allow(private_bounds)]
pub struct SessionTask<'a, E: ElemType, ED: Device, TD: Device> {
    model: &'a Model<E, ED, TD>,
    tokens: Vec<u32>,
    decoded_cursor: usize,
    abort_flag: Arc<AtomicBool>,
    finished_flag: Arc<AtomicBool>,
    token_rx: mpsc::Receiver<u32>,
    kv_rx: mpsc::Receiver<Result<Vec<KVCache<E, TD::Base>>, crate::Error>>,
}

#[allow(private_bounds)]
impl<'a, E: ElemType, ED: Device, TD: Device> SessionTask<'a, E, ED, TD>
where
    ED::Base: DeviceOps<E>,
    TD::Base: DeviceOps<E>,
{
    /// Returns true once the worker thread has stopped.
    pub fn is_finished(&self) -> bool {
        self.finished_flag.load(Ordering::Relaxed)
    }

    /// Stops background work and converts this task back into a reusable `Session`.
    pub fn finish_decoding(mut self) -> Result<Session<'a, E, ED, TD>, crate::Error> {
        self.abort_flag.store(true, Ordering::Relaxed);

        while let Ok(token_id) = self.token_rx.try_recv() {
            self.tokens.push(token_id);
        }
        let tokens = std::mem::take(&mut self.tokens);

        let kv_caches = self
            .kv_rx
            .recv()
            .expect("`Receiver::recv` should succeed")?;

        Ok(Session {
            model: self.model,
            tokens,
            kv_caches,
        })
    }

    /// Returns at most one generated token id as a readable string.
    pub fn get_next_string(&mut self) -> Option<String> {
        while let Some(token_id) = self.token_rx.try_recv().ok() {
            self.tokens.push(token_id);
        }
        if self.tokens.len() <= self.decoded_cursor {
            return None;
        }

        if let Some(special_str) = render_special_token(self.tokens[self.decoded_cursor]) {
            self.decoded_cursor += 1;
            return Some(special_str);
        }

        let un_decoded_tokens = &self.tokens[self.decoded_cursor..];

        if let Ok((consumed, decoded_str)) = self.model.detokenize(un_decoded_tokens) {
            if consumed > 0 {
                self.decoded_cursor += consumed;
                return Some(decoded_str);
            }
        }

        Some(String::new())
    }
}

impl<'a, E: ElemType, ED: Device, TD: Device> Drop for SessionTask<'a, E, ED, TD> {
    fn drop(&mut self) {
        self.abort_flag.store(true, Ordering::Release);
    }
}

fn render_special_token(token: u32) -> Option<String> {
    Some(match token {
        special_token::BEGIN_OF_SENTENCE => "<|begin_of_sentence|>".to_string(),
        special_token::USER => "<|User|>".to_string(),
        special_token::ASSISTANT => "<|Assistant|>".to_string(),
        special_token::THINK_START => "<think>".to_string(),
        special_token::THINK_END => "</think>".to_string(),
        special_token::END_OF_SENTENCE => "<|end_of_sentence|>".to_string(),
        _ => return None,
    })
}

fn run<'a, E: ElemType, ED: Device, TD: Device>(
    model: &'a Model<E, ED, TD>,
    tokens: Vec<u32>,
    mut kv_caches: Vec<KVCache<E, TD::Base>>,
    abort_flag: Arc<AtomicBool>,
    transmitter: mpsc::Sender<u32>,
) -> Result<Vec<KVCache<E, TD::Base>>, crate::Error>
where
    ED::Base: DeviceOps<E>,
    TD::Base: DeviceOps<E>,
    ED: Device<Base = TD::Base>,
{
    let next_token = model.decode(&mut kv_caches, &tokens)?;
    if transmitter.send(next_token).is_err() {
        return Ok(kv_caches);
    }
    if next_token == special_token::END_OF_SENTENCE {
        return Ok(kv_caches);
    }

    let mut current_token = next_token;
    loop {
        if abort_flag.load(Ordering::Relaxed) {
            return Ok(kv_caches);
        }

        let next_token = model.decode(&mut kv_caches, &[current_token])?;
        if transmitter.send(next_token).is_err() {
            return Ok(kv_caches);
        }
        if next_token == special_token::END_OF_SENTENCE {
            return Ok(kv_caches);
        }

        current_token = next_token;
    }
}
