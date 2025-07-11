"use client";

import { FormEvent, useEffect, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import ChatSidebar from "@/components/ChatSidebar";
import ChatWindow from "@/components/ChatWindow";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export interface ChatInstance {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
}

export default function ChatApp() {
  const [chatInstances, setChatInstances] = useState<ChatInstance[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const activeChat = chatInstances.find((chat) => chat.id === activeChatId);

  // Load chats on mount
  useEffect(() => {
    fetch("/api/chats/")
      .then((res) => res.json())
      .then((data: ChatInstance[]) => {
        setChatInstances(data);
        if (data.length > 0) setActiveChatId(data[0].id);
      });
  }, []);

  const createNewChat = async () => {
    const res = await fetch("/api/chats/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "New Chat" }),
    });
    const newChat = await res.json();
    setChatInstances((prev) => [{ ...newChat, messages: [] }, ...prev]);
    setActiveChatId(newChat.id);
  };

  const deleteChat = async (id: string) => {
    await fetch(`/api/chats/${id}`, { method: "DELETE" });

    setChatInstances((prev) => prev.filter((c) => c.id !== id));

    if (activeChatId === id) {
      const remaining = chatInstances.filter((c) => c.id !== id);
      setActiveChatId(remaining[0]?.id || null);
    }
  };

  const sendMessage = async (e: FormEvent<HTMLFormElement>) => {
    e?.preventDefault();
    if (!input.trim() || !activeChat) return;

    const userMessage: Message = {
      id: uuidv4(),
      role: "user",
      content: input.trim(),
    };

    const assistantMessage: Message = {
      id: uuidv4(),
      role: "assistant",
      content: "",
    };

    // Optimistic UI update
    setChatInstances((prev) =>
      prev.map((chat) =>
        chat.id === activeChat.id
          ? {
              ...chat,
              messages: [...chat.messages, userMessage, assistantMessage],
            }
          : chat,
      ),
    );

    setInput("");
    setIsLoading(true);

    const res = await fetch("/api/chat/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [...activeChat.messages, userMessage],
      }),
    });

    const reader = res.body?.getReader();
    const decoder = new TextDecoder();
    let streamedContent = "";

    if (!reader) {
      setIsLoading(false);
      return;
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      streamedContent += chunk;

      // Live update streamed message content
      setChatInstances((prev) =>
        prev.map((chat) =>
          chat.id === activeChat.id
            ? {
                ...chat,
                messages: chat.messages.map((msg) =>
                  msg.id === assistantMessage.id
                    ? { ...msg, content: streamedContent }
                    : msg,
                ),
              }
            : chat,
        ),
      );
    }

    // Save to DB
    await fetch(`/api/chats/${activeChat.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [
          userMessage,
          { ...assistantMessage, content: streamedContent },
        ],
      }),
    });

    setIsLoading(false);
  };

  return (
    <div className="flex h-screen bg-gray-50">
      <ChatSidebar
        chatInstances={chatInstances}
        activeChatId={activeChatId}
        onCreateChat={createNewChat}
        onDeleteChat={deleteChat}
        onSelectChat={setActiveChatId}
      />
      <ChatWindow
        input={input}
        isLoading={isLoading}
        messages={activeChat?.messages || []}
        title={activeChat?.title || ""}
        onSend={sendMessage}
        onInputChange={(value: string) => setInput(value)}
      />
    </div>
  );
}
