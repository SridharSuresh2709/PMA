#!/usr/bin/env python3
"""
inspect_lobes.py
Utility to inspect and visualize data stored in brain lobe collections.
"""

import chromadb
from pathlib import Path
from datetime import datetime

CHROMA_PERSIST_DIR = Path("chroma_db")

def inspect_all_lobes():
    """
    Inspect all brain lobe collections and display their contents.
    """
    print("=" * 80)
    print("🔍 BRAIN LOBE MEMORY INSPECTOR")
    print("=" * 80)
    
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    except Exception as e:
        print(f"❌ Error connecting to ChromaDB: {e}")
        return
    
    # Define expected collections
    expected_lobes = {
        "executive_memory": "🎯 Frontal Lobe (Decision, Planning, Reasoning)",
        "semantic_memory": "📚 Temporal Lobe (Facts, Knowledge, Language)",
        "context_memory": "🌐 Parietal Lobe (Context, Relationships)",
        "visual_memory": "👁️ Occipital Lobe (Visual, Multimodal)",
        "affective_memory": "❤️ Limbic System (Emotion, Importance)",
    }
    
    # Get all collections
    all_collections = client.list_collections()
    print(f"\n📊 Total Collections Found: {len(all_collections)}")
    print("-" * 80)
    
    total_items = 0
    
    for coll_name, description in expected_lobes.items():
        print(f"\n{description}")
        print(f"Collection: {coll_name}")
        print("-" * 80)
        
        try:
            collection = client.get_collection(name=coll_name)
            count = collection.count()
            total_items += count
            
            print(f"📦 Total Items: {count}")
            
            if count > 0:
                # Get all items (limit to 100 for display)
                results = collection.get(limit=min(count, 100))
                
                documents = results.get('documents', [])
                metadatas = results.get('metadatas', [])
                ids = results.get('ids', [])
                
                print(f"\n📝 Sample Entries (showing up to 10):")
                for i, (doc, metadata, doc_id) in enumerate(zip(documents[:10], metadatas[:10], ids[:10])):
                    print(f"\n  Entry #{i+1}")
                    print(f"  ID: {doc_id}")
                    print(f"  Text: {doc[:200]}{'...' if len(doc) > 200 else ''}")
                    
                    if metadata:
                        print(f"  Metadata:")
                        for key, value in metadata.items():
                            if key == "timestamp":
                                try:
                                    dt = datetime.fromisoformat(value)
                                    print(f"    • {key}: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                                except:
                                    print(f"    • {key}: {value}")
                            else:
                                print(f"    • {key}: {value}")
                
                if count > 10:
                    print(f"\n  ... and {count - 10} more entries")
            else:
                print("  ⚠️ No items stored yet")
                
        except Exception as e:
            print(f"  ❌ Error accessing collection: {e}")
    
    print("\n" + "=" * 80)
    print(f"📊 SUMMARY: {total_items} total items across all lobes")
    print("=" * 80)


def inspect_specific_lobe(lobe_name: str):
    """
    Inspect a specific brain lobe collection in detail.
    """
    print("=" * 80)
    print(f"🔍 INSPECTING: {lobe_name}")
    print("=" * 80)
    
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        collection = client.get_collection(name=lobe_name)
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    count = collection.count()
    print(f"\n📦 Total Items: {count}")
    
    if count == 0:
        print("⚠️ No items in this collection")
        return
    
    # Get all items
    results = collection.get()
    documents = results.get('documents', [])
    metadatas = results.get('metadatas', [])
    ids = results.get('ids', [])
    
    print(f"\n📝 ALL ENTRIES:\n")
    for i, (doc, metadata, doc_id) in enumerate(zip(documents, metadatas, ids)):
        print(f"{'=' * 80}")
        print(f"Entry #{i+1}")
        print(f"ID: {doc_id}")
        print(f"\nText:\n{doc}")
        
        if metadata:
            print(f"\nMetadata:")
            for key, value in metadata.items():
                if key == "timestamp":
                    try:
                        dt = datetime.fromisoformat(value)
                        print(f"  • {key}: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    except:
                        print(f"  • {key}: {value}")
                else:
                    print(f"  • {key}: {value}")
        print()


def search_across_lobes(query: str):
    """
    Search for a query across all brain lobe collections.
    """
    print("=" * 80)
    print(f"🔍 SEARCHING: '{query}'")
    print("=" * 80)
    
    from sentence_transformers import SentenceTransformer
    
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    lobes = ["executive_memory", "semantic_memory", "context_memory", 
             "visual_memory", "affective_memory"]
    
    query_emb = embed_model.encode(query, convert_to_numpy=True).tolist()
    
    for lobe_name in lobes:
        try:
            collection = client.get_collection(name=lobe_name)
            count = collection.count()
            
            if count == 0:
                continue
            
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=min(3, count)
            )
            
            docs = results.get('documents', [[]])[0]
            distances = results.get('distances', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            
            if docs:
                print(f"\n📍 {lobe_name.upper().replace('_', ' ')}")
                print("-" * 80)
                
                for i, (doc, dist, md) in enumerate(zip(docs, distances, metadatas)):
                    relevance = 1 - dist
                    print(f"\n  Result #{i+1} (Relevance: {relevance:.3f})")
                    print(f"  Text: {doc[:150]}{'...' if len(doc) > 150 else ''}")
                    if md:
                        role = md.get('role', 'unknown')
                        print(f"  Role: {role}")
        
        except Exception as e:
            print(f"  ❌ Error querying {lobe_name}: {e}")
    
    print("\n" + "=" * 80)


def show_statistics():
    """
    Show statistics about memory distribution across lobes.
    """
    print("=" * 80)
    print("📊 BRAIN LOBE STATISTICS")
    print("=" * 80)
    
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    lobes = {
        "executive_memory": "🎯 Executive",
        "semantic_memory": "📚 Semantic",
        "context_memory": "🌐 Context",
        "visual_memory": "👁️ Visual",
        "affective_memory": "❤️ Affective",
    }
    
    stats = []
    total = 0
    
    for lobe_name, icon_name in lobes.items():
        try:
            collection = client.get_collection(name=lobe_name)
            count = collection.count()
            stats.append((icon_name, count))
            total += count
        except:
            stats.append((icon_name, 0))
    
    print(f"\nTotal Memories: {total}\n")
    
    if total > 0:
        # Sort by count
        stats.sort(key=lambda x: x[1], reverse=True)
        
        max_count = max(s[1] for s in stats)
        
        for name, count in stats:
            percentage = (count / total * 100) if total > 0 else 0
            bar_length = int(count / max_count * 40) if max_count > 0 else 0
            bar = "█" * bar_length
            
            print(f"{name:20} │ {bar:40} │ {count:4} ({percentage:5.1f}%)")
    else:
        print("⚠️ No memories stored yet")
    
    print("\n" + "=" * 80)


def main():
    """
    Interactive menu for inspecting brain lobe collections.
    """
    while True:
        print("\n" + "=" * 80)
        print("🧠 BRAIN LOBE MEMORY INSPECTOR - MENU")
        print("=" * 80)
        print("1. 📊 View all lobes (overview)")
        print("2. 🔍 Inspect specific lobe (detailed)")
        print("3. 🔎 Search across all lobes")
        print("4. 📈 Show statistics")
        print("5. 🚪 Exit")
        print("-" * 80)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            inspect_all_lobes()
        
        elif choice == "2":
            print("\nAvailable lobes:")
            print("  • executive_memory")
            print("  • semantic_memory")
            print("  • context_memory")
            print("  • visual_memory")
            print("  • affective_memory")
            lobe = input("\nEnter lobe name: ").strip()
            inspect_specific_lobe(lobe)
        
        elif choice == "3":
            query = input("\nEnter search query: ").strip()
            if query:
                search_across_lobes(query)
        
        elif choice == "4":
            show_statistics()
        
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()