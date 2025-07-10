import uvicorn
import signal
import sys
import atexit

# Global shutdown flag
server_shutting_down = False

def signal_handler(signum, frame):
    global server_shutting_down
    print("\nShutting down server gracefully...")
    server_shutting_down = True
    
    # Cleanup ML libraries
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    try:
        import numba
        numba.core.dispatcher._dispatcher_cache.clear()
    except:
        pass
    
    sys.exit(0)

def cleanup_on_exit():
    global server_shutting_down
    if not server_shutting_down:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

if __name__ == "__main__":
    # Register signal handlers and cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_on_exit)
    
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        print("Server shutdown complete") 