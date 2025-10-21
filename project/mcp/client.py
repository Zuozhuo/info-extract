import asyncio
from fastmcp import Client

client = Client("http://127.0.0.1:8000/mcp")  # 本地
# client = Client("http://www.neusym.cn:5001/mcp")  # 远程

async def main():
    async with client:
        # 可选：探活
        await client.ping()

        # 列出工具
        tools = await client.list_tools()
        print("tools:", [t.name for t in tools])

        # 调用工具
        result = await client.call_tool("add", {"a": 3, "b": 5})
        print("add result:", result.data)  # 8

        # 调用工具
        result = await client.call_tool("extract_info_from_pdf", {"pdf_path": "/home/zuozhuo/info-extract/output/核材料文档3/auto/核材料文档3_origin.pdf"})
        print("add result:", result.data)  # 8

asyncio.run(main())
