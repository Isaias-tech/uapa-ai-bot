import { useEffect, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { ChatInstance, Message } from "@/types/chat";
import { fetchChats, createChat, streamAIResponse } from "@/lib/api";

export function useChatManager() {
  const [chatInstances, setChatInstances] = useState<ChatInstance[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const activeChat = chatInstances.find((c) => c.id === activeChatId);

  useEffect(() => {
    fetchChats().then((data) => {
      setChatInstances(data);
      setActiveChatId(data[0]?.id ?? null);
    });
  }, []);

  const createNewChat = async () => {
    const chat = await createChat();
    setChatInstances((prev) => [chat, ...prev]);
    setActiveChatId(chat.id);
  };

  const deleteChat = (chatId: string) => {
    setChatInstances((prev) => prev.filter((c) => c.id !== chatId));
    if (chatId === activeChatId) {
      const remaining = chatInstances.filter((c) => c.id !== chatId);
      setActiveChatId(remaining[0]?.id ?? null);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || !activeChat) return;

    const userMsg: Message = {
      id: uuidv4(),
      role: "user",
      content: input.trim(),
    };

    const prevMessages = [...(activeChat?.messages ?? []), userMsg];

    setChatInstances((prev) =>
      prev.map((chat) =>
        chat.id === activeChat.id ? { ...chat, messages: prevMessages } : chat,
      ),
    );

    setInput("");
    setIsLoading(true);

    const reader = await streamAIResponse(prevMessages);
    if (!reader) return;

    const decoder = new TextDecoder();
    let accumulated = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      accumulated += decoder.decode(value, { stream: true });

      setChatInstances((prev) =>
        prev.map((chat) =>
          chat.id === activeChat.id
            ? {
                ...chat,
                messages: [
                  ...prevMessages,
                  {
                    id: uuidv4(),
                    role: "assistant",
                    content: accumulated,
                  },
                ],
              }
            : chat,
        ),
      );
    }

    setIsLoading(false);
  };

  return {
    chatInstances,
    activeChat,
    activeChatId,
    input,
    isLoading,
    setInput,
    createNewChat,
    deleteChat,
    setActiveChatId,
    handleSend,
  };
}
