#!/usr/bin/env python3
"""
Cleanup Service for Temporary Files
Handles cleanup of temporary files without using FastAPI background tasks
"""

import os
import time
import threading
from typing import List
import logging

logger = logging.getLogger(__name__)

class CleanupService:
    """Service to handle cleanup of temporary files"""
    
    def __init__(self):
        self.pending_files: List[str] = []
        self.cleanup_thread = None
        self.should_stop = False
        
    def schedule_cleanup(self, file_path: str, delay_seconds: int = 300):
        """Schedule a file for cleanup after delay_seconds"""
        self.pending_files.append((file_path, time.time() + delay_seconds))
        
        # Start cleanup thread if not already running
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.start_cleanup_thread()
    
    def start_cleanup_thread(self):
        """Start the background cleanup thread"""
        self.should_stop = False
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.info("🧹 Cleanup service started")
    
    def stop_cleanup_thread(self):
        """Stop the background cleanup thread"""
        self.should_stop = True
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        logger.info("🧹 Cleanup service stopped")
    
    def _cleanup_worker(self):
        """Background worker that cleans up files"""
        while not self.should_stop:
            try:
                current_time = time.time()
                files_to_clean = []
                remaining_files = []
                
                # Check which files are ready for cleanup
                for file_path, cleanup_time in self.pending_files:
                    if current_time >= cleanup_time:
                        files_to_clean.append(file_path)
                    else:
                        remaining_files.append((file_path, cleanup_time))
                
                # Clean up ready files
                for file_path in files_to_clean:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.info(f"🗑️ Cleaned up temporary file: {file_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to cleanup file {file_path}: {e}")
                
                # Update pending files list
                self.pending_files = remaining_files
                
                # Sleep for a short period before checking again
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Cleanup worker error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def cleanup_now(self, file_path: str):
        """Immediately cleanup a specific file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"🗑️ Immediately cleaned up file: {file_path}")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to immediately cleanup file {file_path}: {e}")
        return False
    
    def cleanup_all_pending(self):
        """Cleanup all pending files immediately"""
        files_cleaned = 0
        for file_path, _ in self.pending_files:
            if self.cleanup_now(file_path):
                files_cleaned += 1
        
        self.pending_files = []
        logger.info(f"🗑️ Cleaned up {files_cleaned} pending files")
        return files_cleaned

# Global cleanup service instance
cleanup_service = CleanupService()

def get_cleanup_service() -> CleanupService:
    """Get the global cleanup service instance"""
    return cleanup_service

def schedule_file_cleanup(file_path: str, delay_seconds: int = 300):
    """Schedule a file for cleanup after delay_seconds"""
    cleanup_service.schedule_cleanup(file_path, delay_seconds)

def cleanup_file_now(file_path: str):
    """Immediately cleanup a specific file"""
    return cleanup_service.cleanup_now(file_path)

# Initialize cleanup service when module is imported
cleanup_service.start_cleanup_thread() 