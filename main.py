import asyncio
import os
import json
from fastapi import FastAPI, Request, WebSocket
from deepgram.extensions.types.sockets import ListenV1ControlMessage
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from deepgram import AsyncDeepgramClient
from transcription_analyzer import TranscriptionAnalyzer, ContentType

load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Initialize Deepgram Client
api_key = os.getenv("DEEPGRAM_API_KEY")

# Initialize Transcription Analyzer
analyzer = TranscriptionAnalyzer()

@app.get("/", response_class=HTMLResponse)
def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/analysis")
async def get_analysis():
    """Get current analysis of transcriptions"""
    try:
        with open("transcriptions.txt", "r", encoding="utf-8") as f:
            text = f.read()
        
        results = await analyzer.segment_text(text)
        
        return {
            "status": "success",
            "summary": {
                "total_filler": len(results.filler),
                "total_administration": len(results.administration),
                "total_concepts": len(results.visual_concept),
                "total_segments": len(results.filler) + len(results.administration) + len(results.visual_concept)
            },
            "categories": results.to_dict()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/stats")
async def get_stats():
    """Get quick statistics"""
    try:
        with open("transcriptions.txt", "r", encoding="utf-8") as f:
            text = f.read()
        
        results = await analyzer.segment_text(text)
        total = len(results.filler) + len(results.administration) + len(results.visual_concept)
        
        return {
            "filler": {
                "count": len(results.filler),
                "percentage": round((len(results.filler) / total * 100), 2) if total > 0 else 0
            },
            "administration": {
                "count": len(results.administration),
                "percentage": round((len(results.administration) / total * 100), 2) if total > 0 else 0
            },
            "visual_concept": {
                "count": len(results.visual_concept),
                "percentage": round((len(results.visual_concept) / total * 100), 2) if total > 0 else 0
            },
            "total": total
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/buffer_status")
async def get_buffer_status():
    """Get current buffer status"""
    try:
        status = analyzer.get_buffer_status()
        concepts_status = analyzer.get_visual_concepts_status()
        return {
            "status": "success",
            "buffer_status": status,
            "visual_concepts": concepts_status
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket accepted")
    
    deepgram = AsyncDeepgramClient(api_key=api_key)
    background_tasks = set()  # Track background tasks for cleanup
    is_connected = True  # Connection state flag

    try:
        async with deepgram.listen.v1.connect(
            model="nova-2", 
            smart_format="true",
            language="en-US"
        ) as dg_connection:
            print("Deepgram socket connected")

            async def safe_send_json(data):
                """Safely send JSON to the client WebSocket, returns False if connection is dead."""
                nonlocal is_connected
                if not is_connected:
                    return False
                try:
                    await websocket.send_json(data)
                    return True
                except Exception:
                    is_connected = False
                    return False

            async def process_classification(transcript):
                """Background task: classify transcript and send results to client"""
                try:
                    batch_results = await analyzer.add_to_buffer(transcript)
                    
                    if batch_results:
                        print(f"Buffer full! Processing {len(batch_results)} classified segments")
                        
                        for segment_text, content_type in batch_results:
                            # Save to file
                            with open("transcriptions.txt", "a") as f:
                                f.write(segment_text + " ")
                            
                            # Collect visual concepts for image generation
                            if content_type == ContentType.VISUAL_CONCEPT:
                                analyzer.add_visual_concept(segment_text)
                                print(f"  Visual concept added: {segment_text[:60]}...")
                            
                            # Send classification result to client
                            if not await safe_send_json({
                                "type": "classification",
                                "text": segment_text,
                                "category": content_type.value
                            }):
                                return  # WebSocket closed, stop processing
                        
                        # Check if enough visual concepts for image generation
                        if analyzer.should_generate_image():
                            task = asyncio.create_task(generate_and_send_image())
                            background_tasks.add(task)
                            task.add_done_callback(background_tasks.discard)
                    else:
                        # Buffer not full yet — send status
                        buffer_status = analyzer.get_buffer_status()
                        concepts_status = analyzer.get_visual_concepts_status()
                        print(f"Buffering: {buffer_status['buffered_segments']}/{buffer_status['buffer_size']} segments | Concepts: {concepts_status['count']}/{concepts_status['min_required']}")
                        
                        await safe_send_json({
                            "type": "buffering",
                            "buffered": buffer_status['buffered_segments'],
                            "buffer_size": buffer_status['buffer_size'],
                            "concepts_count": concepts_status['count'],
                            "concepts_required": concepts_status['min_required']
                        })
                
                except Exception as e:
                    print(f"Background classification error: {e}")
                    try:
                        rate_status = analyzer.get_rate_limit_status()
                        await safe_send_json({
                            "type": "error",
                            "error": str(e),
                            "rate_limit": rate_status
                        })
                    except Exception:
                        pass

            async def generate_and_send_image():
                """Background task: generate image from visual concepts"""
                try:
                    concepts_status = analyzer.get_visual_concepts_status()
                    print(f"Generating image from {concepts_status['count']} visual concepts...")
                    
                    # Notify client that image generation is starting
                    await safe_send_json({
                        "type": "image_generating",
                        "concepts": concepts_status['concepts']
                    })
                    
                    result = await analyzer.generate_image_from_concepts()
                    if result:
                        print("Image generated! Sending to client...")
                        await safe_send_json({
                            "type": "generated_image",
                            "image_data": result['image_base64'],
                            "mime_type": result['mime_type'],
                            "concepts_used": result['concepts_used']
                        })
                except Exception as e:
                    print(f"Image generation error: {e}")
                    try:
                        rate_status = analyzer.get_rate_limit_status()
                        await safe_send_json({
                            "type": "error",
                            "error": str(e),
                            "rate_limit": rate_status
                        })
                    except Exception:
                        pass

            async def sender():
                """Receive audio/control messages from client and forward audio to Deepgram"""
                nonlocal is_connected
                try:
                    while is_connected:
                        message = await websocket.receive()
                        
                        # Handle text messages (control commands from client)
                        if "text" in message:
                            try:
                                data = json.loads(message["text"])
                                if data.get("type") == "disconnect":
                                    print("Client requested disconnect")
                                    is_connected = False
                                    return
                                elif data.get("type") == "pong":
                                    # Client responded to our ping — connection is alive
                                    pass
                            except (json.JSONDecodeError, KeyError):
                                pass
                        
                        # Handle binary messages (audio data)
                        elif "bytes" in message:
                            audio_data = message["bytes"]
                            if audio_data and len(audio_data) > 0:
                                if hasattr(dg_connection, 'send_media'):
                                    await dg_connection.send_media(audio_data)
                                else:
                                    await dg_connection._send(audio_data)
                        
                        # Handle WebSocket disconnect
                        elif message.get("type") == "websocket.disconnect":
                            print("Client WebSocket disconnected")
                            is_connected = False
                            return
                            
                except Exception as e:
                    print(f"Sender stopped: {e}")
                    is_connected = False

            async def keepalive():
                """Send keepalive to Deepgram to prevent 10s inactivity timeout"""
                try:
                    while is_connected:
                        await asyncio.sleep(5)
                        if not is_connected:
                            break
                        await dg_connection.send_control(
                            ListenV1ControlMessage(type="KeepAlive")
                        )
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"Keepalive error: {e}")

            async def heartbeat():
                """Send periodic ping to client to keep WebSocket alive"""
                try:
                    while is_connected:
                        await asyncio.sleep(10)
                        if not is_connected:
                            break
                        if not await safe_send_json({"type": "ping"}):
                            break
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"Heartbeat error: {e}")

            async def receiver():
                """Receive transcripts from Deepgram and dispatch to background tasks"""
                try:
                    async for message in dg_connection:
                        if not is_connected:
                            break
                        try:
                            if hasattr(message, 'channel'):
                                alternatives = message.channel.alternatives
                                if alternatives and len(alternatives) > 0:
                                    transcript = alternatives[0].transcript
                                    if transcript:
                                        print(f"Transcript: {transcript}")
                                        
                                        # Send live transcript to client immediately
                                        if not await safe_send_json({
                                            "type": "transcript",
                                            "text": transcript
                                        }):
                                            break
                                        
                                        # Fire off classification as a background task
                                        task = asyncio.create_task(
                                            process_classification(transcript)
                                        )
                                        background_tasks.add(task)
                                        task.add_done_callback(background_tasks.discard)
                        
                        except Exception as e:
                            print(f"Error processing message (continuing): {e}")
                            continue
                        
                except Exception as e:
                    print(f"Receiver error: {e}")

            # Run all tasks concurrently
            sender_task = asyncio.create_task(sender())
            receiver_task = asyncio.create_task(receiver())
            keepalive_task = asyncio.create_task(keepalive())
            heartbeat_task = asyncio.create_task(heartbeat())

            done, pending = await asyncio.wait(
                [sender_task, receiver_task, keepalive_task, heartbeat_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Log which task finished first
            for task in done:
                exc = task.exception() if not task.cancelled() else None
                if exc:
                    print(f"Task failed with exception: {exc}")
                else:
                    print(f"Task completed normally")

            # Signal all tasks to stop
            is_connected = False
            for task in pending:
                task.cancel()
            
            # Wait briefly for pending tasks to finish
            if pending:
                await asyncio.wait(pending, timeout=3.0)

    except Exception as e:
        print(f"Error in websocket_endpoint: {e}")
        import traceback
        traceback.print_exc()
    finally:
        is_connected = False
        # Cancel any remaining background tasks
        for task in background_tasks:
            task.cancel()
        print("WebSocket closed")