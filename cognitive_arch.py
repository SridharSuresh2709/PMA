#!/usr/bin/env python3
"""
cognitive_arch.py
Voice+Text chatbot with brain lobe-inspired multi-collection ChromaDB memory.
Features:
- Automatic topic name generation from first user message using Gemini
- Brain lobe memory collections (executive, semantic, context, visual, affective)
- Working memory buffer with auto-consolidation
"""

import os
import time
import uuid
import threading
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone
from collections import deque
from typing import Dict, List, Optional
import re

# --------- Configuration ---------
SESSIONS_DIR = Path("sessions")
CHROMA_PERSIST_DIR = Path("chroma_db")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3  # Per collection
GEMINI_MODEL = "gemini-2.0-flash"
WORKING_MEMORY_SIZE = 10  # Number of items before consolidation

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

import chromadb
from sentence_transformers import SentenceTransformer
import speech_recognition as sr
import pyttsx3


# ----------------- Brain Lobe Memory Store -----------------
class BrainLobeMemoryStore:
    """
    Multi-collection memory system inspired by brain lobes.
    Each collection represents a specialized memory type.
    """
    
    def __init__(self):
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        except Exception:
            settings = chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(CHROMA_PERSIST_DIR),
            )
            self.client = chromadb.Client(settings=settings)

        # Initialize embedding model
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.lock = threading.Lock()
        
        # Create brain lobe collections
        self.collections = {
            # Frontal Lobe: Decision-making, planning, reasoning
            "executive_memory": self.client.get_or_create_collection(
                name="executive_memory",
                metadata={"description": "Decisions, plans, reasoning traces, goals"}
            ),
            
            # Temporal Lobe: Language, facts, semantic knowledge
            "semantic_memory": self.client.get_or_create_collection(
                name="semantic_memory",
                metadata={"description": "Facts, definitions, long-term declarative knowledge"}
            ),
            
            # Parietal Lobe: Context integration, spatial-temporal relationships
            "context_memory": self.client.get_or_create_collection(
                name="context_memory",
                metadata={"description": "Session context, time, location, task relationships"}
            ),
            
            # Occipital Lobe: Visual processing
            "visual_memory": self.client.get_or_create_collection(
                name="visual_memory",
                metadata={"description": "Visual data, images, diagrams, screenshots"}
            ),
            
            # Limbic System: Emotional tagging, importance
            "affective_memory": self.client.get_or_create_collection(
                name="affective_memory",
                metadata={"description": "Emotional weight, importance scores, priorities"}
            ),
        }
        
        # Working memory buffer (Hippocampus function)
        self.working_memory = deque(maxlen=WORKING_MEMORY_SIZE)
        
    def _classify_memory_type(self, text: str, metadata: Dict) -> str:
        """
        Classify which brain lobe collection this memory belongs to.
        Simple keyword-based classification (can be enhanced with ML).
        """
        text_lower = text.lower()
        role = metadata.get("role", "")
        
        # Check for executive/planning keywords
        executive_keywords = ["plan", "decide", "should", "will", "goal", "strategy", 
                            "think", "reason", "next", "step", "project", "work on"]
        if any(kw in text_lower for kw in executive_keywords):
            return "executive_memory"
        
        # Check for factual/semantic keywords
        semantic_keywords = ["is", "are", "means", "definition", "fact", "what", 
                           "who", "where", "when", "explain", "called", "system"]
        if any(kw in text_lower for kw in semantic_keywords):
            return "semantic_memory"
        
        # Check for emotional/importance keywords
        affective_keywords = ["important", "urgent", "love", "hate", "feel", 
                            "emotion", "priority", "critical", "remember this"]
        if any(kw in text_lower for kw in affective_keywords):
            return "affective_memory"
        
        # Check for visual keywords
        visual_keywords = ["image", "picture", "visual", "see", "look", "diagram", "chart"]
        if any(kw in text_lower for kw in visual_keywords):
            return "visual_memory"
        
        # Default: context memory for general conversational context
        return "context_memory"
    
    def add_to_working_memory(self, text: str, source_file=None, metadata=None):
        """
        Add item to working memory buffer (Hippocampus).
        Consolidates to long-term storage when buffer is full.
        """
        if metadata is None:
            metadata = {}
        if source_file:
            metadata["source"] = str(source_file)
        
        metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        item = {
            "text": text,
            "metadata": metadata,
            "embedding": None  # Computed during consolidation
        }
        
        self.working_memory.append(item)
        
        # Auto-consolidate if buffer is full
        if len(self.working_memory) >= WORKING_MEMORY_SIZE:
            self.consolidate_working_memory()
    
    def consolidate_working_memory(self):
        """
        Transfer items from working memory to appropriate long-term collections.
        This mimics the hippocampus consolidation process.
        """
        if not self.working_memory:
            return
        
        print(f"\n[Consolidating {len(self.working_memory)} items from working memory...]")
        
        for item in self.working_memory:
            text = item["text"]
            metadata = item["metadata"]
            
            # Classify which collection this belongs to
            collection_name = self._classify_memory_type(text, metadata)
            
            # Add to appropriate long-term collection
            self.add_to_collection(text, collection_name, metadata)
        
        # Clear working memory
        self.working_memory.clear()
        print("[Consolidation complete]")
    
    def add_to_collection(self, text: str, collection_name: str, metadata=None):
        """
        Add a single text item to a specific brain lobe collection.
        """
        if metadata is None:
            metadata = {}
        
        if collection_name not in self.collections:
            print(f"Warning: Unknown collection {collection_name}, using context_memory")
            collection_name = "context_memory"
        
        metadata["collection"] = collection_name
        
        # Generate embedding
        emb = self.embed_model.encode(text, convert_to_numpy=True).tolist()
        uid = str(uuid.uuid4())
        
        with self.lock:
            self.collections[collection_name].add(
                documents=[text],
                embeddings=[emb],
                ids=[uid],
                metadatas=[metadata],
            )
    
    def query_collection(self, text: str, collection_name: str, top_k: int = TOP_K) -> List[Dict]:
        """
        Query a specific brain lobe collection.
        """
        if collection_name not in self.collections:
            return []
        
        q_emb = self.embed_model.encode(text, convert_to_numpy=True).tolist()
        
        with self.lock:
            results = self.collections[collection_name].query(
                query_embeddings=[q_emb],
                n_results=top_k,
            )
        
        hits = []
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        for doc, dist, md in zip(docs, distances, metadatas):
            hits.append({
                "text": doc,
                "score": float(dist),
                "metadata": md,
                "collection": collection_name
            })
        
        return hits
    
    def query_all_lobes(self, text: str, top_k_per_lobe: int = TOP_K) -> Dict[str, List[Dict]]:
        """
        Query all brain lobe collections and return organized results.
        """
        results = {}
        
        for collection_name in self.collections.keys():
            hits = self.query_collection(text, collection_name, top_k_per_lobe)
            if hits:
                results[collection_name] = hits
        
        return results
    
    def get_consolidated_context(self, text: str, prioritize_collections: List[str] = None) -> str:
        """
        Get consolidated context from all relevant brain lobes.
        Optionally prioritize certain collections.
        """
        if prioritize_collections is None:
            prioritize_collections = ["executive_memory", "semantic_memory", "context_memory"]
        
        all_results = self.query_all_lobes(text)
        
        context_parts = []
        
        # Add prioritized collections first
        for coll_name in prioritize_collections:
            if coll_name in all_results and all_results[coll_name]:
                context_parts.append(f"\n[{coll_name.upper().replace('_', ' ')}]")
                for hit in all_results[coll_name][:TOP_K]:
                    md = hit.get("metadata", {}) or {}
                    src = md.get("source", "unknown")
                    context_parts.append(f"  • {hit['text']} (relevance: {1-hit['score']:.2f})")
        
        # Add remaining collections
        for coll_name, hits in all_results.items():
            if coll_name not in prioritize_collections and hits:
                context_parts.append(f"\n[{coll_name.upper().replace('_', ' ')}]")
                for hit in hits[:TOP_K]:
                    md = hit.get("metadata", {}) or {}
                    context_parts.append(f"  • {hit['text']} (relevance: {1-hit['score']:.2f})")
        
        if not context_parts:
            return "No relevant memories found in any brain lobe."
        
        return "\n".join(context_parts)


# ----------------- Gemini client wrapper -----------------
def generate_with_gemini(prompt, model=GEMINI_MODEL, temperature=0.7, max_tokens=512):
    """Generate response using Google Gemini."""
    try:
        import google.generativeai as genai
    except Exception:
        return "Gemini client not installed. pip install google-generativeai"

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("VERTEX_API_KEY")
    if not api_key:
        return "Gemini API key not found. Set GOOGLE_API_KEY or VERTEX_API_KEY env var."

    try:
        genai.configure(api_key=api_key)
    except Exception:
        try:
            genai.api_key = api_key
        except Exception:
            pass

    try:
        model_client = genai.GenerativeModel(model)
        response = model_client.generate_content(prompt)
        if hasattr(response, "text"):
            return response.text
        return str(response)
    except Exception as e:
        return f"Gemini call failed: {e}"


def generate_topic_name(first_message: str) -> str:
    """
    Generate a concise topic name from the user's first message using Gemini.
    Returns a short, descriptive filename-safe topic name.
    """
    prompt = f"""Given this first message from a user, generate a concise topic name (2-4 words max) that captures the main subject.

User's first message: "{first_message}"

Requirements:
- 2-4 words maximum
- Lowercase with underscores instead of spaces
- No special characters except underscores
- Descriptive and meaningful
- Examples: "project_planning", "python_learning", "memory_assistant", "travel_ideas"

Respond ONLY with the topic name, nothing else."""

    try:
        topic = generate_with_gemini(prompt, temperature=0.3, max_tokens=20)
        
        # Clean up the response
        topic = topic.strip().lower()
        # Remove quotes if present
        topic = topic.strip('"\'')
        # Replace spaces with underscores
        topic = re.sub(r'[\s\-]+', '_', topic)
        # Remove any non-alphanumeric characters except underscores
        topic = re.sub(r'[^\w_]', '', topic)
        # Limit length
        topic = topic[:50]
        
        # Fallback if generation failed
        if not topic or len(topic) < 3:
            topic = "conversation"
        
        return topic
        
    except Exception as e:
        print(f"⚠️  Error generating topic name: {e}")
        return "conversation"


# ----------------- Voice & TTS helpers -----------------
def listen_from_mic(timeout=5, phrase_time_limit=10):
    """Listen to microphone input and convert to text."""
    if sr is None:
        raise RuntimeError("SpeechRecognition not installed. pip install SpeechRecognition and pyaudio.")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... speak now.")
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        return r.recognize_google(audio)
    except Exception as e:
        raise RuntimeError(f"Speech recognition failed: {e}")

def speak_text(text):
    """Convert text to speech."""
    if pyttsx3 is None:
        print("(TTS not available) " + text)
        return
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("(TTS error)", e)

# ----------------- Session file helpers -----------------
def new_session_file(topic_name: str = None) -> Path:
    """
    Create a new session file with optional topic name.
    If topic_name is provided, format: topic_YYYYMMDDTHHMMSSZ.txt
    Otherwise: session_YYYYMMDDTHHMMSSZ.txt
    """
    SESSIONS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    if topic_name:
        fname = SESSIONS_DIR / f"{topic_name}_{ts}.txt"
    else:
        fname = SESSIONS_DIR / f"session_{ts}.txt"
    
    fname.touch(exist_ok=False)
    return fname

def append_to_session(file_path: Path, role: str, text: str):
    """Append a message to the session file."""
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {role}: {text}\n"
    file_path.open("a", encoding="utf-8").write(line)

# ----------------- Inspection helpers -----------------
def inspect_lobes(store: BrainLobeMemoryStore):
    """Display overview of all brain lobe collections."""
    print("\n" + "=" * 70)
    print("🔍 BRAIN LOBE INSPECTION")
    print("=" * 70)
    
    lobe_icons = {
        "executive_memory": "🎯",
        "semantic_memory": "📚",
        "context_memory": "🌐",
        "visual_memory": "👁️",
        "affective_memory": "❤️",
    }
    
    for lobe_name, collection in store.collections.items():
        icon = lobe_icons.get(lobe_name, "📦")
        count = collection.count()
        
        print(f"\n{icon} {lobe_name.upper().replace('_', ' ')}")
        print(f"   Items: {count}")
        
        if count > 0:
            # Get a few sample items
            results = collection.get(limit=3)
            docs = results.get('documents', [])
            
            print(f"   Recent samples:")
            for i, doc in enumerate(docs[:3], 1):
                preview = doc[:80] + "..." if len(doc) > 80 else doc
                print(f"     {i}. {preview}")
    
    print("\n" + "=" * 70)

def show_lobe_statistics(store: BrainLobeMemoryStore):
    """Show statistics about memory distribution."""
    print("\n" + "=" * 70)
    print("📊 MEMORY DISTRIBUTION STATISTICS")
    print("=" * 70)
    
    stats = []
    total = 0
    
    lobe_names = {
        "executive_memory": "🎯 Executive",
        "semantic_memory": "📚 Semantic",
        "context_memory": "🌐 Context",
        "visual_memory": "👁️ Visual",
        "affective_memory": "❤️ Affective",
    }
    
    for lobe_name, collection in store.collections.items():
        count = collection.count()
        stats.append((lobe_names.get(lobe_name, lobe_name), count))
        total += count
    
    print(f"\n📦 Total Memories: {total}")
    print(f"🧠 Working Memory Buffer: {len(store.working_memory)}/{WORKING_MEMORY_SIZE}")
    print()
    
    if total > 0:
        stats.sort(key=lambda x: x[1], reverse=True)
        max_count = max(s[1] for s in stats) if stats else 1
        
        for name, count in stats:
            percentage = (count / total * 100) if total > 0 else 0
            bar_length = int(count / max_count * 40) if max_count > 0 else 0
            bar = "█" * bar_length
            
            print(f"{name:15} │{bar:42}│ {count:4} ({percentage:5.1f}%)")
    else:
        print("⚠️  No memories stored in long-term collections yet")
    
    print("\n" + "=" * 70)

def search_lobes(store: BrainLobeMemoryStore, query: str):
    """Search across all lobes and display results."""
    print("\n" + "=" * 70)
    print(f"🔎 SEARCHING: '{query}'")
    print("=" * 70)
    
    results = store.query_all_lobes(query, top_k_per_lobe=2)
    
    if not results:
        print("\n⚠️  No results found")
        return
    
    lobe_icons = {
        "executive_memory": "🎯",
        "semantic_memory": "📚",
        "context_memory": "🌐",
        "visual_memory": "👁️",
        "affective_memory": "❤️",
    }
    
    for lobe_name, hits in results.items():
        icon = lobe_icons.get(lobe_name, "📦")
        print(f"\n{icon} {lobe_name.upper().replace('_', ' ')}")
        
        for i, hit in enumerate(hits, 1):
            relevance = 1 - hit['score']
            text_preview = hit['text'][:100] + "..." if len(hit['text']) > 100 else hit['text']
            print(f"   {i}. [{relevance:.2%} match] {text_preview}")
    
    print("\n" + "=" * 70)

# ----------------- Main Loop -----------------
def main():
    print("=" * 70)
    print("🧠 BRAIN LOBE MEMORY SYSTEM")
    print("=" * 70)
    print("Multi-collection ChromaDB with brain-inspired memory lobes:")
    print("  • Executive Memory (Frontal): Decisions, planning, reasoning")
    print("  • Semantic Memory (Temporal): Facts, knowledge, definitions")
    print("  • Context Memory (Parietal): Session context, relationships")
    print("  • Visual Memory (Occipital): Visual/multimodal data")
    print("  • Affective Memory (Limbic): Emotional weight, importance")
    print("  • Working Memory (Hippocampus): Short-term consolidation buffer")
    print("=" * 70)
    
    store = BrainLobeMemoryStore()
    
    # Session file will be created after first message with auto-generated topic
    session_file = None
    first_message = True
    
    print("\nReady! Commands:")
    print("  • 'voice' - Use voice input")
    print("  • 'inspect' - View all brain lobes")
    print("  • 'stats' - Show memory statistics")
    print("  • 'search: <query>' - Search across lobes")
    print("  • 'consolidate' - Force memory consolidation")
    print("  • 'exit' - Quit")
    print("-" * 70)

    while True:
        user_input = input("\n💬 You: ").strip()
        if not user_input:
            continue
        
        if user_input.lower() in ("exit", "quit"):
            # Consolidate remaining working memory before exit
            store.consolidate_working_memory()
            print("\n👋 Goodbye!")
            break

        if user_input.lower() == "consolidate":
            store.consolidate_working_memory()
            continue
        
        if user_input.lower() == "inspect":
            inspect_lobes(store)
            continue
        
        if user_input.lower() == "stats":
            show_lobe_statistics(store)
            continue
        
        if user_input.lower().startswith("search:"):
            query = user_input[7:].strip()
            search_lobes(store, query)
            continue

        if user_input.lower() == "voice":
            try:
                user_text = listen_from_mic()
                print(f"🎤 You (voice): {user_text}")
            except Exception as e:
                print(f"❌ Voice error: {e}")
                continue
        else:
            user_text = user_input

        # Generate topic name and create session file on first message
        if first_message:
            print("🤔 Generating topic name from your message...")
            topic_name = generate_topic_name(user_text)
            session_file = new_session_file(topic_name)
            print(f"📝 Topic: '{topic_name}'")
            print(f"📄 Session file: {session_file}")
            first_message = False

        # Save to session file
        append_to_session(session_file, "user", user_text)

        # Add to working memory (will auto-consolidate when full)
        try:
            store.add_to_working_memory(
                f"User: {user_text}",
                source_file=session_file,
                metadata={"role": "user"}
            )
        except Exception as e:
            print(f"❌ Error adding to working memory: {e}")

        # Get consolidated context from all brain lobes
        context = store.get_consolidated_context(user_text)

        system_prompt = (
            "You are an AI assistant with a brain lobe-inspired memory system. "
            "You have access to specialized memory collections (executive, semantic, context, visual, affective). "
            "Use the following memories to provide an informed, contextual response.\n\n"
            f"MEMORIES FROM BRAIN LOBES:\n{context}\n\n"
            f"USER QUERY: {user_text}\n\n"
            "Provide a concise, helpful response based on relevant memories."
        )

        # Generate reply with Gemini
        reply = generate_with_gemini(system_prompt)
        if not reply:
            reply = "Sorry, I couldn't generate a reply."

        # Save assistant reply
        append_to_session(session_file, "assistant", reply)
        try:
            store.add_to_working_memory(
                f"Assistant: {reply}",
                source_file=session_file,
                metadata={"role": "assistant"}
            )
        except Exception as e:
            print(f"❌ Error adding assistant reply to working memory: {e}")

        # Display and speak
        print(f"\n🤖 Assistant: {reply}")
        try:
            speak_text(reply)
        except Exception:
            pass

if __name__ == "__main__":
    main()