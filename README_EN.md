Note: This is an unofficial modified version of the KouriChat project.  

The main changes in this version involve reverting the conversation memory system from the original post-v1.3.9 model ("short-term memory + 10-round core memory refresh") back to a model similar to the older versions (v1.3.7 and earlier), which follows a "short-term memory (summarized upon reaching a threshold) + accumulative long-term memory" approach.  

**Key Changes:**  

- Instead of refreshing core memory every 10 rounds, this version triggers a summary when short-term memory reaches a certain length (~15 rounds/30 lines of dialogue). The new summary is appended to the existing core memory rather than overwriting it. The goal is to achieve longer-lasting and less easily lost conversation memory.  

**Implementation Details:**  
- The primary modifications were made to the `modules/memory/memory_service.py` file.  
- To maintain as much compatibility as possible with the newer code structure, the original function names (e.g., `update_core_memory`, `get_core_memory`) were retained, but their internal logic was replaced with the older summarization, appending, and retrieval (partially simulated) functions.  

**Important Note:**  
- The `get_core_memory` function now returns the most recently generated memory summary from the long-term memory buffer. This is a compromise for compatibility and does not fully replicate the older behavior of retrieving the most relevant memory on demand (the relevant retrieval logic has been added as the `get_relevant_memories` method but requires invocation from other parts of the project to take effect).  
- This modification may be incompatible with future official updates of KouriChat.  

Original project repository: https://github.com/KouriChat/KouriChat