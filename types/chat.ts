export interface Message {
  id: string;
  role: "user" | "assistant" | "streaming";
  content: string;
}

export interface ChatInstance {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
}
