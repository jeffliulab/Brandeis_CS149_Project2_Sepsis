import sys 
from pathlib import Path 
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from app.config import AppConfig
from app.logger import logger
import asyncio


# Local application imports
try:
    from open_manus.app.agent.manus import Manus
    from open_manus.app.logger import logger as manus_logger
except ImportError as e:
    print(f"Warning: Unable to import open_manus module: {e}")
    # Create a mock Manus class to avoid runtime errors
    class Manus:
        async def run(self, prompt):
            print(f"Simulating Manus execution: {prompt}")
            return f"Simulated Manus result: {prompt}"
        
        async def cleanup(self):
            print("Simulating cleanup of Manus resources")



# =====================================================================
# Tools Processing Module
# =====================================================================
class ToolsProcessor:
    """
    Class for handling tool-related content in LLM responses
    
    This class is responsible for extracting and processing TOOLS instructions
    from LLM responses and determining subsequent actions based on instruction status
    """
    
    @staticmethod
    def extract_tools_content(message):
        """Extract TOOLS instruction content from a message"""
        # Use regex to match TOOLS instruction format
        import re
        # pattern = r'\[\[TOOLS:(TRUE|FALSE)\]\[(.*?)\]\]'
        pattern = r'\[\[TOOLS:(TRUE|FALSE|SEPSIS)\]\[(.*?)\]\]'
        match = re.search(pattern, message, re.DOTALL)
        
        if match:
            raw_tool     = match.group(1)              # "TRUE"、"FALSE" 或 "SEPSIS"
            tools_status  = (raw_tool == "TRUE")
            # 对 SEPSIS 指令，我们把 tools_content 也设置成 "SEPSIS"
            tools_content = raw_tool if raw_tool == "SEPSIS" else match.group(2)

            
            # Debug output to verify content extraction
            logger.info(f"Extracted tool status: {tools_status}")
            logger.info(f"Extracted tool content: {tools_content}")
            
            # Based on HIDE_TOOLS_CONTENT, decide whether to remove TOOLS instruction from message
            if AppConfig.HIDE_TOOLS_CONTENT:
                # Remove the matched TOOLS instruction part
                cleaned_message = re.sub(pattern, '', message, flags=re.DOTALL).strip()
            else:
                # Keep original message
                cleaned_message = message
                
            return cleaned_message, tools_status, tools_content
        
        # If no TOOLS instruction matched, return original message and default values
        logger.warning("No TOOLS instruction content detected")
        return message, False, ""
    
    @staticmethod
    async def process_tools_request_async_with_progress(content, progress_callback=None):
        """
        Asynchronously process tool requests with progress reporting
        
        Parameters:
            content (str): Tool request content to pass to Manus
            progress_callback (callable): Function to call with progress updates
            
        Returns:
            str: Tool execution result
        """
        logger.info(f"Starting tool request processing: {content}")

        # —— 新增：如果是 SEPSIS 指令，直接运行本地管道脚本 —— 
        if content == "SEPSIS":
            import subprocess
            from pathlib import Path
            import sys
            import time
            
            root = Path(__file__).resolve().parent.parent.parent
            cmds = [
                "python -m sepsis.1_load",
                "python -m sepsis.2_impute",
                "python -m sepsis.3_feature",
                "python -m sepsis.4_train",
                "python -m sepsis.5_evaluate",
                "python -m sepsis.6_explain",
                "python -m sepsis.7_predict",
            ]
            logs = []
            
            # 添加进度显示
            total_steps = len(cmds)
            
            for i, c in enumerate(cmds, 1):
                step_name = c.split(".")[-1]
                progress_msg = f"[{i}/{total_steps}] 运行: {c} ({step_name})..."
                
                # 在控制台显示进度
                print(progress_msg, file=sys.stderr)
                sys.stderr.flush()
                
                # 记录开始时间
                start_time = time.time()
                
                # 运行命令并捕获输出
                proc = subprocess.run(c.split(), cwd=str(root), capture_output=True, text=True)
                
                # 计算运行时间
                elapsed_time = time.time() - start_time
                
                # 显示命令执行结果状态
                status = "成功" if proc.returncode == 0 else "失败"
                status_msg = f"[{i}/{total_steps}] {step_name} {status} (耗时: {elapsed_time:.2f}秒)"
                print(status_msg, file=sys.stderr)
                sys.stderr.flush()
                
                # 实时显示输出
                if proc.stdout:
                    print(f"--- 标准输出 ---\n{proc.stdout}", file=sys.stderr)
                    sys.stderr.flush()
                if proc.stderr:
                    print(f"--- 错误输出 ---\n{proc.stderr}", file=sys.stderr)
                    sys.stderr.flush()
                
                # 添加到日志
                logs.append(f"$ {c}\n状态: {status} (耗时: {elapsed_time:.2f}秒)\n--- 输出 ---\n{proc.stdout}{proc.stderr}")
                
                # 如果命令执行失败，中止流程
                if proc.returncode != 0:
                    print(f"命令执行失败，停止流程: {c}", file=sys.stderr)
                    sys.stderr.flush()
                    break
            
            print("所有流程执行完毕", file=sys.stderr)
            sys.stderr.flush()
            
            return "\n\n".join(logs)
            # —— 新增部分结束 ——
                
        try:
            # Dynamically import Manus
            try:
                from open_manus.app.agent.manus import Manus
            except ImportError as e:
                logger.error(f"Unable to import Manus: {e}")
                if progress_callback:
                    progress_callback(f"Failed to import Manus module: {e}")
                return f"Tool initialization failed: Cannot import Manus module ({e})"
            
            # Create Manus agent instance
            agent = Manus()
            
            # Set progress callback if provided
            if progress_callback:
                agent.set_progress_callback(progress_callback)
                progress_callback("Manus agent initialized")
            
            try:
                # Ensure prompt is not empty
                if not content.strip():
                    logger.warning("Received empty tool request")
                    if progress_callback:
                        progress_callback("Tool request content is empty")
                    return "Tool request content is empty, cannot process"
                
                logger.info(f"Submitting request to Open Manus: {content}")
                if progress_callback:
                    progress_callback(f"Starting execution with prompt: {content[:100]}...")
                
                # Run the agent with the provided content
                result = await agent.run(content)
                
                logger.info("Open Manus toolchain execution completed")
                if progress_callback:
                    progress_callback("Tool execution completed successfully")
                
                return result
                
            except Exception as e:
                logger.error(f"Error executing Open Manus toolchain: {e}")
                if progress_callback:
                    progress_callback(f"Error during execution: {str(e)}")
                return f"Tool execution failed: {str(e)}"
            finally:
                # Ensure resources are cleaned up
                logger.info("Cleaning up Open Manus resources")
                if progress_callback:
                    progress_callback("Cleaning up resources")
                await agent.cleanup()
                
        except Exception as e:
            logger.error(f"Error initializing Open Manus agent: {e}")
            if progress_callback:
                progress_callback(f"Tool initialization error: {str(e)}")
            return f"Tool initialization failed: {str(e)}"
        finally:
            # Restore original logging configuration
            if progress_callback:
                progress_callback("Resource cleanup and logging restoration complete")
    
    @staticmethod
    def process_tools_request(content):
        """Synchronous wrapper function for processing tool requests with progress"""
        # Record received tool request content
        logger.info(f"Received tool request: {content}")
        
        # Use thread pool executor to run asynchronous function
        import concurrent.futures
        import threading
        
        logger.info("Creating thread pool executor")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Create a shared result container
            result_container = []
            progress_updates = []
            
            # Progress callback function
            def progress_handler(message):
                progress_updates.append(message)
                logger.info(f"Progress update: {message}")
            
            # Define thread function
            def run_async_in_thread():
                try:
                    logger.info("Thread execution started")
                    # Create new event loop
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    
                    # Execute asynchronous function with progress
                    result = new_loop.run_until_complete(
                        ToolsProcessor.process_tools_request_async_with_progress(
                            content, progress_callback=progress_handler
                        )
                    )
                    result_container.append(result)
                    logger.info("Asynchronous task completed")

                except Exception as e:
                    logger.error(f"Thread execution error: {e}")
                    result_container.append(f"Tool processing error: {str(e)}")
                finally:
                    logger.info("Thread ended")
            
            # Submit thread task
            future = executor.submit(run_async_in_thread)
            
            try:
                # Wait for task completion, set timeout
                logger.info("Waiting for tool execution to complete, timeout 300 seconds")
                future.result(timeout=300)
                
                # If there's a result, return it with progress info; otherwise return default message
                if result_container:
                    progress_summary = "\n".join(progress_updates) if progress_updates else "No detailed progress available"
                    return f"""
                            Tool execution completed.

                            Progress Log:
                            {progress_summary}

                            Result:
                            {result_container[0]}
                            """
                else:
                    return "Tool execution completed, but no result returned"
            except concurrent.futures.TimeoutError:
                logger.error("Tool execution timeout")
                progress_summary = "\n".join(progress_updates) if progress_updates else "No progress information available"
                return f"""
                        Tool execution timeout, forcibly terminated after 300 seconds.

                        Last recorded progress:
                        {progress_summary}
                        """
            except Exception as e:
                logger.error(f"Error waiting for tool execution result: {e}")
                progress_summary = "\n".join(progress_updates) if progress_updates else "No progress information available"
                return f"""
                        Error during tool processing: {str(e)}

                        Progress before error:
                        {progress_summary}
                        """
    
    @staticmethod
    def process_message(message):
        """
        Process complete LLM response message
        
        Parameters:
            message (str): Original LLM response message
            
        Returns:
            str: Processed message
        """
        # Log original message for debugging
        logger.info(f"Processing original message: {message[:100]}..." if len(message) > 100 else f"Processing original message: {message}")
        
        # Extract TOOLS instruction content
        cleaned_message, tools_status, tools_content = ToolsProcessor.extract_tools_content(message)
        
        # Determine subsequent processing based on TOOLS status
        # if tools_status:
        #     # If TOOLS status is TRUE, process tool request and add result to message
        #     tools_result = ToolsProcessor.process_tools_request(tools_content)
        #     # Build final response, adding tool execution result after cleaned message
        #     final_message = f"{cleaned_message}\n\n[Tool Execution Result]: {tools_result}"
        #     return final_message
        # else:
        #     # If TOOLS status is FALSE, just return cleaned message
        #     return cleaned_message
        if tools_content == "SEPSIS":
            import subprocess
            root = Path(__file__).resolve().parent.parent.parent
            cmds = [
                "python -m sepsis.1_load",
                "python -m sepsis.2_impute",
                "python -m sepsis.3_feature",
                "python -m sepsis.4_train",
                "python -m sepsis.5_evaluate",
                "python -m sepsis.6_explain",
                "python -m sepsis.7_predict",
            ]
            logs = []
            for c in cmds:
                proc = subprocess.run(c.split(), cwd=str(root), capture_output=True, text=True)
                logs.append(f"$ {c}\n{proc.stdout}{proc.stderr}")
                if proc.returncode != 0:
                    break 
            return cleaned_message + "\n\n[Sepsis Pipeline Logs]\n" + "\n".join(logs)
        
        if tools_status:
            tools_result = ToolsProcessor.process_tools_request(tools_content)
            return f"{cleaned_message}\n\n[Tool Execution Result]: {tools_result}"
        return cleaned_message
    
    
    


