import os
import json
from helpers.extension import Extension
from agent import LoopData


class VerifyFaiss(Extension):
    """Verify FAISS index integrity before memory recall runs.
    
    Checks that index.faiss and index.pkl exist in the active memory
    directory. If missing or corrupted, auto-creates a fresh empty
    index using the embedding model specified in embedding.json.
    
    This prevents ValueError crashes in the recall extension when
    FAISS files are lost (e.g., after storage migration or corruption).
    """

    _checked = False  # class-level flag: only run once per agent lifetime

    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not self.agent:
            return

        # Only check once per agent boot, not every monologue
        if VerifyFaiss._checked:
            return
        VerifyFaiss._checked = True

        memory_dir = "/a0/usr/memory/default"
        index_faiss = os.path.join(memory_dir, "index.faiss")
        index_pkl = os.path.join(memory_dir, "index.pkl")
        embedding_json = os.path.join(memory_dir, "embedding.json")

        # Check if both index files exist and are non-empty
        faiss_ok = os.path.isfile(index_faiss) and os.path.getsize(index_faiss) > 0
        pkl_ok = os.path.isfile(index_pkl) and os.path.getsize(index_pkl) > 0

        if faiss_ok and pkl_ok:
            # Files exist and are non-empty — do a quick load test
            try:
                import pickle
                with open(index_pkl, "rb") as f:
                    data = pickle.load(f)
                # If we get here without error, index is loadable
                self.agent.context.log.log(
                    type="info",
                    heading="FAISS integrity check",
                    content="FAISS index OK.",
                )
                return
            except Exception as e:
                self.agent.context.log.log(
                    type="warning",
                    heading="FAISS integrity check",
                    content=f"FAISS index.pkl corrupted: {e}. Rebuilding...",
                )
        else:
            missing = []
            if not faiss_ok:
                missing.append("index.faiss")
            if not pkl_ok:
                missing.append("index.pkl")
            self.agent.context.log.log(
                type="warning",
                heading="FAISS integrity check",
                content=f"Missing FAISS files: {', '.join(missing)}. Rebuilding...",
            )

        # Rebuild: determine embedding model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"  # safe default
        if os.path.isfile(embedding_json):
            try:
                with open(embedding_json, "r") as f:
                    emb_config = json.load(f)
                model_name = emb_config.get("model_name", model_name)
            except Exception:
                pass

        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            faiss_store = FAISS.from_texts(["initialization placeholder"], embeddings)

            os.makedirs(memory_dir, exist_ok=True)
            faiss_store.save_local(memory_dir)

            self.agent.context.log.log(
                type="info",
                heading="FAISS integrity check",
                content=f"Fresh FAISS index created at {memory_dir} using {model_name}.",
            )
        except Exception as e:
            self.agent.context.log.log(
                type="error",
                heading="FAISS integrity check",
                content=f"Failed to rebuild FAISS index: {e}",
            )
