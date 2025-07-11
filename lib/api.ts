import { ChatInstance, Message } from "@/types/chat";

export const fetchChats = async (): Promise<ChatInstance[]> => {
  const res = await fetch("/api/chats/");
  return res.json();
};

export const createChat = async (): Promise<ChatInstance> => {
  const res = await fetch("/api/chats/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "New Chat" }),
  });
  return res.json();
};

export const streamAIResponse = async (messages: Message[]) => {
  const res = await fetch("/api/chat/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });

  return res.body?.getReader();
};
