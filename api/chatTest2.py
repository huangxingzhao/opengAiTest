import openai

MODEL = "text-davinci-003"

openai.api_key = "sk-bneyQXujVwm4zZKX1f6e77228a344e3d826a8fD5F431Dc13"
openai.api_base = "https://api.rcouyi.com/v1"


class Conversation:
    def __init__(self, prompt, num_of_round):
        self.prompt = prompt
        self.num_of_round = num_of_round
        self.messages = []
        self.messages.append({"role": "system", "content": self.prompt})

    def ask(self, question):
        try:
          self.messages.append({"role": "user", "content": question})
          response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=self.messages,
          temperature=0.5,
          max_tokens=2048,
          top_p=1,
                )


        except Exception as e:
            print(e)
            return e

        message = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round*2 + 1:
            del self.messages[1:3]
        return message

prompt = """ 你是一个中国初始，用中文回答做菜问题，要满足以下要求：
    1.回答必须中文
    2.回答限制在100个字以内

"""
conv1 = Conversation(prompt, 2)
question1 = "你是谁？"
print("User : %s" % question1)
print("Assistant : %s\n" % conv1.ask(question1))

question2 = "请问鱼香肉丝怎么做？"
print("User : %s" % question2)
print("Assistant : %s\n" % conv1.ask(question2))

question3 = "那蚝油牛肉呢？"
print("User : %s" % question3)
print("Assistant : %s\n" % conv1.ask(question3))

