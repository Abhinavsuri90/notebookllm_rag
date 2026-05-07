/**
 * NotebookLM RAG — Server
 *
 * Full RAG pipeline:
 *   Ingestion → Chunking → Embedding → Storage → Retrieval → Generation
 *
 * Chunking Strategy: Recursive Character Text Splitter
 *   - Splits text on natural boundaries (paragraphs → sentences → words)
 *   - Uses a chunk size of 1000 characters with 200 character overlap
 *   - Overlap ensures context is not lost at chunk boundaries
 *   - Each chunk retains metadata (page number, source file) for citation
 */

import "dotenv/config";
import express from "express";
import multer from "multer";
import cors from "cors";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { v4 as uuidv4 } from "uuid";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { OpenAI } from "openai";

// ── Paths ────────────────────────────────────────────────────────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ── Express setup ────────────────────────────────────────────────────────────
const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// ── File upload setup ────────────────────────────────────────────────────────
const uploadsDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadsDir),
  filename: (_req, file, cb) => {
    const uniqueName = `${uuidv4()}-${file.originalname}`;
    cb(null, uniqueName);
  },
});

const upload = multer({
  storage,
  fileFilter: (_req, file, cb) => {
    const allowed = [".pdf", ".txt"];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowed.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error("Only PDF and TXT files are supported."));
    }
  },
  limits: { fileSize: 20 * 1024 * 1024 }, // 20 MB
});

// ── Shared instances (lazy-loaded) ───────────────────────────────────────────
let _embeddings = null;
let _openai = null;

function getEmbeddings() {
  if (!_embeddings) {
    const config = { model: "openai/text-embedding-3-large" };
    // Support OpenRouter or any custom OpenAI-compatible endpoint
    if (process.env.OPENAI_BASE_URL) {
      config.configuration = { baseURL: process.env.OPENAI_BASE_URL };
    }
    _embeddings = new OpenAIEmbeddings(config);
  }
  return _embeddings;
}

function getOpenAI() {
  if (!_openai) {
    const config = {};
    if (process.env.OPENAI_BASE_URL) {
      config.baseURL = process.env.OPENAI_BASE_URL;
    }
    _openai = new OpenAI(config);
  }
  return _openai;
}

// ── In-memory document registry ─────────────────────────────────────────────
// Maps collectionName → { originalName, uploadedAt, chunkCount }
const documentRegistry = new Map();

// ═══════════════════════════════════════════════════════════════════════════════
//  CHUNKING STRATEGY — Recursive Character Text Splitter
// ═══════════════════════════════════════════════════════════════════════════════
//
//  Why Recursive Character Text Splitter?
//  1. It tries to split on the most semantically meaningful boundary first
//     (double newline → single newline → sentence-ending punctuation → space).
//  2. If a chunk is still too large it falls back to smaller boundaries.
//  3. Overlap ensures no information is lost between adjacent chunks.
//
//  Parameters:
//    chunkSize   = 1000  — each chunk ≤ 1000 characters
//    chunkOverlap = 200  — consecutive chunks share 200 characters of context
// ═══════════════════════════════════════════════════════════════════════════════

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Load a file and return raw LangChain Document objects.
 */
async function loadFile(filePath, originalName) {
  const ext = path.extname(originalName).toLowerCase();

  if (ext === ".pdf") {
    const loader = new PDFLoader(filePath);
    return loader.load();
  }

  // Plain text
  const text = fs.readFileSync(filePath, "utf-8");
  return [
    {
      pageContent: text,
      metadata: { source: originalName, page: 1 },
    },
  ];
}

/**
 * Chunk documents using the Recursive Character Text Splitter.
 */
async function chunkDocuments(docs) {
  const chunks = await textSplitter.splitDocuments(docs);
  // Add a sequential chunk index to each chunk's metadata
  return chunks.map((chunk, i) => {
    chunk.metadata.chunkIndex = i;
    return chunk;
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
//  API ROUTES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * POST /api/upload
 * Ingestion endpoint — accepts a PDF or TXT file, chunks it, embeds it,
 * and stores the vectors in Qdrant.
 */
app.post("/api/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded." });
    }

    const filePath = req.file.path;
    const originalName = req.file.originalname;
    // Unique collection per document so multiple docs don't collide
    const collectionName = `doc-${uuidv4()}`;

    console.log(`📄 Ingesting: ${originalName}`);

    // Step 1 — Load
    const docs = await loadFile(filePath, originalName);
    console.log(`   ✔ Loaded ${docs.length} page(s)`);

    // Step 2 — Chunk
    const chunks = await chunkDocuments(docs);
    console.log(`   ✔ Created ${chunks.length} chunk(s)`);

    // Step 3 & 4 — Embed + Store in Qdrant
    const qdrantConfig = {
      collectionName,
    };
    // Support both local Qdrant and Qdrant Cloud
    if (process.env.QDRANT_URL) {
      qdrantConfig.url = process.env.QDRANT_URL;
    }
    if (process.env.QDRANT_API_KEY) {
      qdrantConfig.apiKey = process.env.QDRANT_API_KEY;
    }

    await QdrantVectorStore.fromDocuments(chunks, getEmbeddings(), qdrantConfig);
    console.log(`   ✔ Stored in Qdrant collection "${collectionName}"`);

    // Register the document
    documentRegistry.set(collectionName, {
      originalName,
      uploadedAt: new Date().toISOString(),
      chunkCount: chunks.length,
    });

    // Clean up the uploaded file from disk
    fs.unlinkSync(filePath);

    res.json({
      success: true,
      collectionName,
      originalName,
      chunkCount: chunks.length,
      pageCount: docs.length,
    });
  } catch (err) {
    console.error("Upload error:", err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * POST /api/chat
 * Retrieval + Generation endpoint — takes a user query and a collectionName,
 * retrieves the most relevant chunks, and generates a grounded answer.
 */
app.post("/api/chat", async (req, res) => {
  try {
    const { query, collectionName } = req.body;

    if (!query || !collectionName) {
      return res
        .status(400)
        .json({ error: "Both 'query' and 'collectionName' are required." });
    }

    console.log(`💬 Query: "${query}" → collection: ${collectionName}`);

    // Step 5 — Retrieval: find the top-k most relevant chunks
    const qdrantConfig = {
      collectionName,
    };
    if (process.env.QDRANT_URL) {
      qdrantConfig.url = process.env.QDRANT_URL;
    }
    if (process.env.QDRANT_API_KEY) {
      qdrantConfig.apiKey = process.env.QDRANT_API_KEY;
    }

    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      getEmbeddings(),
      qdrantConfig
    );

    const retriever = vectorStore.asRetriever({ k: 5 });
    const relevantChunks = await retriever.invoke(query);

    console.log(`   ✔ Retrieved ${relevantChunks.length} chunk(s)`);

    // Build context string with page numbers for citation
    const contextBlocks = relevantChunks.map((chunk, i) => {
      const page = chunk.metadata?.loc?.pageNumber ?? chunk.metadata?.page ?? "N/A";
      return `[Chunk ${i + 1} | Page ${page}]\n${chunk.pageContent}`;
    });
    const context = contextBlocks.join("\n\n---\n\n");

    // Step 6 — Generation: send context + query to the LLM
    const systemPrompt = `You are a helpful AI assistant that answers questions ONLY based on the provided document context.

RULES:
1. Answer ONLY using the information in the context below.
2. If the answer is not in the context, say "I couldn't find information about that in the uploaded document."
3. Cite page numbers when possible (e.g., "According to page 3, ...").
4. Be clear, concise, and well-structured. Use bullet points or numbered lists when appropriate.
5. Do NOT use any prior knowledge — only the document context.

DOCUMENT CONTEXT:
${context}`;

    const response = await getOpenAI().chat.completions.create({
      model: "openai/gpt-4.1-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: query },
      ],
      temperature: 0.3, // Low temperature for factual, grounded answers
    });

    const answer = response.choices[0].message.content;
    console.log(`   ✔ Generated answer`);

    res.json({
      answer,
      sources: relevantChunks.map((chunk) => ({
        page: chunk.metadata?.loc?.pageNumber ?? chunk.metadata?.page ?? "N/A",
        preview: chunk.pageContent.substring(0, 200) + "…",
      })),
    });
  } catch (err) {
    console.error("Chat error:", err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/documents
 * Returns a list of all uploaded documents for the current session.
 */
app.get("/api/documents", (_req, res) => {
  const docs = [];
  for (const [collectionName, meta] of documentRegistry) {
    docs.push({ collectionName, ...meta });
  }
  res.json(docs);
});

/**
 * GET /health
 * Health-check endpoint.
 */
app.get("/health", (_req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// ── Start ────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`\n🚀 NotebookLM RAG server running on http://localhost:${PORT}\n`);
});
