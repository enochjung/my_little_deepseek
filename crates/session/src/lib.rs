mod special_token;

use core::{Backend, ElemType, MLTError, MemoryOwn};
use inference::{KVCache, Model};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};

/// Manages the mutable, evolving state of an active inference process.
///
/// While a [`Model`] acts as the static, immutable blueprint, a `Session` holds the contextual memory
/// required for text generation, primarily consisting of the dynamic KV Cache and the
/// historical sequence of processed tokens.
///
/// # Examples
///
/// ```no_run
/// use my_little_deepseek::{Model, Session, config::Configure};
///
/// let config = Configure::new();
/// let model = Model::new(config).unwrap();
/// let mut session = model.new_session().unwrap();
/// ```
pub struct Session<'model, T: ElemType, EB: Backend<T>, TB: Backend<T>> {
    model: &'model Model<T, EB, TB>,
    tokens: Vec<u32>,
    kv_caches: Vec<KVCache<T, TB::Memory>>,
}

impl<'model, T: ElemType, EB: Backend<T>, TB: Backend<T>> Session<'model, T, EB, TB> {
    /// Creates a new, stateful inference session bound to this model.
    ///
    /// The resulting [`Session`] will allocate its own isolated KV Cache and maintain
    /// a unique token history, allowing multiple independent generations to run concurrently
    /// against the same immutable model.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use my_little_deepseek::{Model, config::Configure};
    ///
    /// let config = Configure::new();
    /// let model = Model::new(config).unwrap();
    /// let session = model.new_session().unwrap();
    /// ```
    pub fn new(model: &'model Model<T, EB, TB>) -> Result<Self, MLTError> {
        const INITIAL_N: usize = 1024;

        let tokens = vec![special_token::BEGIN_OF_SENTENCE; 1];
        let kv_caches = (0..model.num_hidden_layers)
            .map(|_| {
                let k_mem = TB::Memory::new(
                    INITIAL_N * model.head_size as usize * model.num_key_value_heads,
                )?;
                let v_mem = TB::Memory::new(
                    INITIAL_N * model.head_size as usize * model.num_key_value_heads,
                )?;
                KVCache::new(k_mem, v_mem, model.head_size, model.num_key_value_heads, 0)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            model,
            tokens,
            kv_caches,
        })
    }

    /// Initiates a background text generation task for the given user prompt.
    ///
    /// This method appends the prompt to the session's token history, applies the necessary
    /// special chat template tokens, and spawns a worker thread within the provided scope.
    /// It consumes the `Session` and returns a [`SessionTask`] used to stream the asynchronous output.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use my_little_deepseek::{Model, config::Configure};
    ///
    /// let config = Configure::new();
    /// let model = Model::new(config).unwrap();
    /// let session = model.new_session().unwrap();
    ///
    /// std::thread::scope(|s| {
    ///     let task = session.send_prompt(s, "Explain quantum computing").unwrap();
    /// });
    /// ```
    pub fn send_prompt<'scope>(
        self,
        scope: &'scope std::thread::Scope<'scope, '_>,
        user_input: &str,
    ) -> Result<SessionTask<'model, T, EB, TB>, MLTError>
    where
        'model: 'scope,
        TB: Backend<T, Memory = EB::Memory>, // TODO
    {
        let Self {
            model,
            mut tokens,
            kv_caches,
        } = self;

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
}

/// An asynchronous handle representing an ongoing inference computation.
///
/// A `SessionTask` streams generated tokens from a background worker thread. It maintains the
/// decoding cursor to ensure valid UTF-8 strings are yielded as the neural network predicts
/// subsequent byte-level tokens.
///
/// # Examples
///
/// ```no_run
/// use my_little_deepseek::{Model, config::Configure};
/// use std::time::Duration;
///
/// let config = Configure::new();
/// let model = Model::new(config).unwrap();
/// let session = model.new_session().unwrap();
///
/// std::thread::scope(|s| {
///     let mut task = session.send_prompt(s, "Hello!").unwrap();
///     
///     while !task.is_finished() {
///         while let Some(text) = task.get_next_string() {
///             print!("{text}");
///         }
///         std::thread::sleep(Duration::from_millis(50));
///     }
///     
///     let _session = task.finish_decoding().unwrap();
/// });
/// ```
pub struct SessionTask<'model, T: ElemType, EB: Backend<T>, TB: Backend<T>> {
    model: &'model Model<T, EB, TB>,
    tokens: Vec<u32>,
    decoded_cursor: usize,
    abort_flag: Arc<AtomicBool>,
    finished_flag: Arc<AtomicBool>,
    token_rx: mpsc::Receiver<u32>,
    kv_rx: mpsc::Receiver<Result<Vec<KVCache<T, TB::Memory>>, MLTError>>,
}

impl<'model, T: ElemType, EB: Backend<T>, TB: Backend<T>> SessionTask<'model, T, EB, TB> {
    /// Returns `true` if the background generation thread has completed its execution.
    ///
    /// This occurs when the model emits a stop token, reaches the maximum sequence length,
    /// or if the generation is manually aborted.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use my_little_deepseek::{Model, config::Configure};
    /// # let config = Configure::new();
    /// # let model = Model::new(config).unwrap();
    /// # let session = model.new_session().unwrap();
    /// # std::thread::scope(|s| {
    /// # let task = session.send_prompt(s, "Hi").unwrap();
    /// if task.is_finished() {
    ///     println!("Generation complete!");
    /// }
    /// # });
    /// ```
    pub fn is_finished(&self) -> bool {
        self.finished_flag.load(Ordering::Relaxed)
    }

    /// Aborts any ongoing computation and reclaims the underlying `Session`.
    ///
    /// This method gracefully halts the worker thread, synchronizes the KV Caches,
    /// and flushes any pending tokens into the history. The returned [`Session`] can then be reused
    /// for subsequent prompts, maintaining the conversational context.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use my_little_deepseek::{Model, config::Configure};
    /// # let config = Configure::new();
    /// # let model = Model::new(config).unwrap();
    /// # let session = model.new_session().unwrap();
    /// # std::thread::scope(|s| {
    /// # let task = session.send_prompt(s, "Hi").unwrap();
    /// // Interrupt or finish the current task
    /// let session = task.finish_decoding().unwrap();
    /// # });
    /// ```
    pub fn finish_decoding(mut self) -> Result<Session<'model, T, EB, TB>, MLTError> {
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

    /// Attempts to retrieve the next chunk of decoded text from the generator.
    ///
    /// Because the model operates on Byte-Pair Encoding, a single generated token might not
    /// represent a complete, valid UTF-8 character. This method buffers raw tokens internally and
    /// only yields a `String` when a valid character boundary is resolved.
    ///
    /// Returns `None` if the generator has not yet produced enough tokens to surpass the
    /// internal decoding cursor, or an empty `String` if the accumulated tokens do not yet
    /// form a complete character.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use my_little_deepseek::{Model, config::Configure};
    /// # let config = Configure::new();
    /// # let model = Model::new(config).unwrap();
    /// # let session = model.new_session().unwrap();
    /// # std::thread::scope(|s| {
    /// # let mut task = session.send_prompt(s, "Hi").unwrap();
    /// if let Some(text) = task.get_next_string() {
    ///     print!("{}", text);
    /// }
    /// # });
    /// ```
    pub fn get_next_string(&mut self) -> Option<String> {
        while let Ok(token_id) = self.token_rx.try_recv() {
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

        if let Ok((consumed, decoded_str)) = self.model.detokenize(un_decoded_tokens)
            && consumed > 0
        {
            self.decoded_cursor += consumed;
            return Some(decoded_str);
        }

        Some(String::new())
    }
}

impl<'a, T: ElemType, EB: Backend<T>, TB: Backend<T>> Drop for SessionTask<'a, T, EB, TB> {
    fn drop(&mut self) {
        self.abort_flag.store(true, Ordering::Release);
    }
}

fn render_special_token(token: u32) -> Option<String> {
    Some(match token {
        special_token::END_OF_SENTENCE => "<|end_of_sentence|>".to_string(),
        special_token::USER => "<|User|>".to_string(),
        special_token::ASSISTANT => "<|Assistant|>".to_string(),
        special_token::BEGIN_OF_SENTENCE => "<|begin_of_sentence|>".to_string(),
        special_token::EOT => "<|EOT|>".to_string(),
        special_token::THINK_START => "<think>".to_string(),
        special_token::THINK_END => "</think>".to_string(),
        special_token::QUAD_START => "<|quad_start|>".to_string(),
        special_token::QUAD_END => "<|quad_end|>".to_string(),
        special_token::VISION_START => "<|vision_start|>".to_string(),
        special_token::VISION_END => "<|vision_end|>".to_string(),
        special_token::VISION_PAD => "<|vision_pad|>".to_string(),
        special_token::IMAGE_PAD => "<|image_pad|>".to_string(),
        special_token::VIDEO_PAD => "<|video_pad|>".to_string(),
        special_token::TOOL_CALL_START => "<tool_call>".to_string(),
        special_token::TOOL_CALL_END => "</tool_call>".to_string(),
        special_token::FIM_PREFIX => "<|fim_prefix|>".to_string(),
        special_token::FIM_MIDDLE => "<|fim_middle|>".to_string(),
        special_token::FIM_SUFFIX => "<|fim_suffix|>".to_string(),
        special_token::FIM_PAD => "<|fim_pad|>".to_string(),
        special_token::REPO_NAME => "<|repo_name|>".to_string(),
        special_token::FILE_SEP => "<|file_sep|>".to_string(),
        _ => return None,
    })
}

fn run<T: ElemType, EB: Backend<T>, TB: Backend<T>>(
    model: &Model<T, EB, TB>,
    tokens: Vec<u32>,
    mut kv_caches: Vec<KVCache<T, TB::Memory>>,
    abort_flag: Arc<AtomicBool>,
    transmitter: mpsc::Sender<u32>,
) -> Result<Vec<KVCache<T, TB::Memory>>, MLTError>
where
    TB: Backend<T, Memory = EB::Memory>, // TODO
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
