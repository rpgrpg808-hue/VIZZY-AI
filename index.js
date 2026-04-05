// Vizzy AI - Professional Backend with Personality & Multi-User
// Secrets: GROQ_API_KEY, GEMINI_API_KEY

const express = require('express');
const cors = require('cors');
const Groq = require('groq-sdk');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// AI Clients
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const geminiModel = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

// ---------- Data Structures ----------
// userData: Map<userId, { conversations: Map<convId, { id, title, messages, summary }>, settings: { personalityPrompt }, nextId }>
const userData = new Map();

function getUserData(userId) {
  if (!userData.has(userId)) {
    userData.set(userId, { 
      conversations: new Map(), 
      nextId: 1,
      settings: { personalityPrompt: 'Sen Vizzy AI, yardımsever ve profesyonel bir asistan. Türkçe cevap ver.' }
    });
  }
  return userData.get(userId);
}

function createConversation(userId, title = 'Yeni Sohbet') {
  const user = getUserData(userId);
  const id = String(user.nextId++);
  const now = new Date().toISOString();
  const conv = {
    id,
    title,
    messages: [],
    summary: '',
    createdAt: now,
    updatedAt: now,
  };
  user.conversations.set(id, conv);
  return conv;
}

function getConversation(userId, convId) {
  const user = getUserData(userId);
  return user.conversations.get(convId);
}

async function updateConversation(userId, convId, userMsg, assistantMsg) {
  const conv = getConversation(userId, convId);
  if (!conv) return null;
  conv.messages.push({ role: 'user', content: userMsg });
  conv.messages.push({ role: 'assistant', content: assistantMsg });
  conv.updatedAt = new Date().toISOString();

  if (conv.messages.length > 30) {
    const oldMessages = conv.messages.slice(0, -20);
    const text = oldMessages.map(m => `${m.role}: ${m.content}`).join('\n').slice(0, 2000);
    conv.summary = `[Önceki konuşma özeti: ${text.slice(0, 500)}...]`;
    conv.messages = conv.messages.slice(-20);
  }
  conv.messages = truncateHistory(conv.messages, 20000);
  return conv;
}

function estimateTokens(text) {
  return Math.ceil((text || '').length / 4);
}

function truncateHistory(messages, maxTokens) {
  let total = 0;
  const truncated = [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const t = estimateTokens(messages[i].content);
    if (total + t > maxTokens) break;
    total += t;
    truncated.unshift(messages[i]);
  }
  return truncated;
}

// ---------- AI Call with Personality & Fallback ----------
async function callAIWithFallback(userId, conv, userMessage, res) {
  const user = getUserData(userId);
  const personalityPrompt = user.settings.personalityPrompt;

  const recentMessages = conv.messages.slice(-15);
  let fullContext = [...recentMessages];
  if (conv.summary) {
    fullContext.unshift({ role: 'system', content: conv.summary });
  }
  fullContext.unshift({ role: 'system', content: personalityPrompt });
  fullContext.push({ role: 'user', content: userMessage });

  fullContext = truncateHistory(fullContext, 28000);
  const messagesForAI = fullContext.map(m => ({
    role: m.role === 'assistant' ? 'assistant' : (m.role === 'user' ? 'user' : 'system'),
    content: m.content
  }));

  try {
    const stream = await groq.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: messagesForAI,
      stream: true,
      temperature: 0.8,
      max_tokens: 4096,
    });
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    let fullResponse = '';
    for await (const chunk of stream) {
      const text = chunk.choices[0]?.delta?.content || '';
      fullResponse += text;
      res.write(text);
    }
    res.end();
    return fullResponse;
  } catch (err) {
    console.error('Groq error, fallback to Gemini:', err.message);
    try {
      const geminiPrompt = messagesForAI.map(m => `${m.role}: ${m.content}`).join('\n');
      const result = await geminiModel.generateContent({
        contents: [{ role: 'user', parts: [{ text: geminiPrompt }] }],
        generationConfig: { temperature: 0.8, maxOutputTokens: 4096 },
      });
      const responseText = result.response.text();
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.write(responseText);
      res.end();
      return responseText;
    } catch (geminiErr) {
      console.error('Gemini also failed:', geminiErr);
      res.status(500).send('Üzgünüm, şu anda hizmet veremiyorum.');
      return '';
    }
  }
}

// ---------- Rate Limiting ----------
const rateLimitStore = new Map();
function rateLimitCheck(ip, userId) {
  const key = `${ip}_${userId}`;
  const now = Date.now();
  const record = rateLimitStore.get(key) || { count: 0, lastReset: now, cooldownUntil: 0 };
  if (record.cooldownUntil > now) {
    return { allowed: false, retryAfter: Math.ceil((record.cooldownUntil - now) / 1000) };
  }
  if (now - record.lastReset > 1000) {
    record.count = 0;
    record.lastReset = now;
  }
  record.count++;
  if (record.count > 10) {
    record.cooldownUntil = now + 2000;
    rateLimitStore.set(key, record);
    return { allowed: false, retryAfter: 2 };
  }
  rateLimitStore.set(key, record);
  return { allowed: true, slowDown: record.count > 6 };
}

// ---------- Queue ----------
const requestQueue = [];
let processingQueue = false;
async function processQueue() {
  if (processingQueue || requestQueue.length === 0) return;
  processingQueue = true;
  while (requestQueue.length > 0) {
    const { resolve, reject, userId, conv, userMessage, res } = requestQueue.shift();
    try {
      const response = await callAIWithFallback(userId, conv, userMessage, res);
      resolve(response);
    } catch (err) {
      reject(err);
    }
    await new Promise(r => setTimeout(r, 50));
  }
  processingQueue = false;
}
function addToQueue(userId, conv, userMessage, res) {
  return new Promise((resolve, reject) => {
    requestQueue.push({ resolve, reject, userId, conv, userMessage, res });
    processQueue();
  });
}

// ---------- Cache ----------
const cacheStore = new Map();
function getCacheKey(message, convId) { return `${convId}_${message.slice(0, 100)}`; }
function getCached(key) {
  const entry = cacheStore.get(key);
  if (entry && Date.now() - entry.timestamp < 60000) return entry.response;
  return null;
}
function setCache(key, response) {
  cacheStore.set(key, { response, timestamp: Date.now() });
  if (cacheStore.size > 100) {
    const oldest = [...cacheStore.entries()].sort((a,b) => a[1].timestamp - b[1].timestamp)[0];
    cacheStore.delete(oldest[0]);
  }
}

// ---------- API Endpoints ----------
// Get all conversations
app.get('/api/conversations', (req, res) => {
  const { userId } = req.query;
  if (!userId) return res.status(400).json({ error: 'userId required' });
  const user = getUserData(userId);
  const convs = Array.from(user.conversations.values()).map(c => ({
    id: c.id,
    title: c.title,
    createdAt: c.createdAt,
    updatedAt: c.updatedAt,
  }));
  res.json(convs);
});

// Create new conversation
app.post('/api/conversations', (req, res) => {
  const { userId, title } = req.body;
  if (!userId) return res.status(400).json({ error: 'userId required' });
  const conv = createConversation(userId, title || 'Yeni Sohbet');
  res.json({ id: conv.id, title: conv.title });
});

// Get conversation messages
app.get('/api/conversations/:id', (req, res) => {
  const { userId } = req.query;
  const { id } = req.params;
  if (!userId) return res.status(400).json({ error: 'userId required' });
  const conv = getConversation(userId, id);
  if (!conv) return res.status(404).json({ error: 'Conversation not found' });
  res.json({ messages: conv.messages, title: conv.title });
});

// Rename conversation
app.put('/api/conversations/:id', (req, res) => {
  const { userId, title } = req.body;
  const { id } = req.params;
  if (!userId || !title) return res.status(400).json({ error: 'userId and title required' });
  const conv = getConversation(userId, id);
  if (!conv) return res.status(404).json({ error: 'Conversation not found' });
  conv.title = title;
  res.json({ success: true });
});

// Delete conversation
app.delete('/api/conversations/:id', (req, res) => {
  const { userId } = req.body;
  const { id } = req.params;
  if (!userId) return res.status(400).json({ error: 'userId required' });
  const user = getUserData(userId);
  if (!user.conversations.has(id)) return res.status(404).json({ error: 'Not found' });
  user.conversations.delete(id);
  res.json({ success: true });
});

// Get user settings (personality)
app.get('/api/settings', (req, res) => {
  const { userId } = req.query;
  if (!userId) return res.status(400).json({ error: 'userId required' });
  const user = getUserData(userId);
  res.json({ personalityPrompt: user.settings.personalityPrompt });
});

// Update user settings (personality)
app.put('/api/settings', (req, res) => {
  const { userId, personalityPrompt } = req.body;
  if (!userId) return res.status(400).json({ error: 'userId required' });
  const user = getUserData(userId);
  if (personalityPrompt !== undefined) {
    user.settings.personalityPrompt = personalityPrompt;
  }
  res.json({ success: true, personalityPrompt: user.settings.personalityPrompt });
});

// Chat endpoint
app.post('/api/chat/:convId', async (req, res) => {
  const { message, userId } = req.body;
  const { convId } = req.params;
  const ip = req.ip || req.socket.remoteAddress;

  if (!message || !userId || !convId) {
    return res.status(400).json({ error: 'Missing fields' });
  }

  const rate = rateLimitCheck(ip, userId);
  if (!rate.allowed) {
    return res.status(429).json({ error: `Çok fazla istek. ${rate.retryAfter} saniye bekleyin.` });
  }
  if (rate.slowDown) await new Promise(r => setTimeout(r, 800));

  let conv = getConversation(userId, convId);
  if (!conv) {
    conv = createConversation(userId, 'Yeni Sohbet');
  }

  const cacheKey = getCacheKey(message, convId);
  const cached = getCached(cacheKey);
  if (cached) {
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    await updateConversation(userId, convId, message, cached);
    return res.send(cached);
  }

  try {
    const response = await addToQueue(userId, conv, message, res);
    await updateConversation(userId, convId, message, response);
    setCache(cacheKey, response);

    if (conv.messages.length === 2 && conv.title === 'Yeni Sohbet') {
      const shortTitle = message.slice(0, 30) + (message.length > 30 ? '…' : '');
      conv.title = shortTitle;
    }
  } catch (err) {
    console.error('Chat error:', err);
    if (!res.headersSent) {
      res.status(500).json({ error: 'AI servisi geçici olarak kullanılamıyor.' });
    }
  }
});

app.get('/health', (req, res) => res.send('OK'));

function startServer(port) {
  const server = app.listen(port, () => console.log(`Vizzy AI running on port ${port}`));
  server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      console.log(`Port ${port} busy, trying ${port+1}`);
      startServer(port+1);
    } else {
      console.error(err);
      process.exit(1);
    }
  });
}
startServer(PORT);
