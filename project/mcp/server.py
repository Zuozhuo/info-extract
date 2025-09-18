from fastmcp import FastMCP

mcp = FastMCP("Demo")

@mcp.tool
def add(a: int, b: int) -> int:
    "Add two numbers"
    return a + b

if __name__ == "__main__":
    # 本地开发如果想走 HTTP，建议改用 CLI 启动（见下）
    # fastmcp run server.py:mcp --transport http --port 8000
    mcp.run()

