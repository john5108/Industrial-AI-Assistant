import requests

# ========== 你的配置信息（不用改）==========
API_KEY = "09658251-47c1-4755-a294-61e63773b9ae"
MODEL = "ep-20260330005736-422fj"
URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
# ======================================

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# 维护对话上下文（保证AI记得之前的对话）
messages = []

print("✅ 已连接豆包AI，输入消息开始对话（输入「退出」结束）\n")
while True:
    # 1. 获取用户输入
    user_input = input("你：")
    if user_input == "退出":
        print("👋 对话结束")
        break
    
    # 2. 把用户消息加入上下文
    messages.append({"role": "user", "content": user_input})
    
    # 3. 发送请求
    data = {
        "model": MODEL,
        "messages": messages
    }
    response = requests.post(URL, headers=headers, json=data)
    result = response.json()
    
    if response.status_code == 200:
        ai_reply = result["choices"][0]["message"]["content"]
        print(f"AI：{ai_reply}\n")
        # 把AI回复也加入上下文，保证下一轮对话连贯
        messages.append({"role": "assistant", "content": ai_reply})
    else:
        print(f"❌ 请求失败：{result}\n")