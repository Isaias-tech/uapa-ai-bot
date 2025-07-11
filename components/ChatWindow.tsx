"use client";

import { Bot, Send, User } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { FormEvent, useRef } from "react";
import { Message } from "@/types/chat";
import ReactMarkdown from "react-markdown";
import MarkdownRenderer from "./MarkdownRenderer";

interface ChatWindowProps {
  title: string;
  messages: Message[];
  input: string;
  isLoading: boolean;
  onInputChange: (value: string) => void;
  onSend: (e: FormEvent<HTMLFormElement>) => void;
}

export default function ChatWindow({
  title,
  messages,
  input,
  isLoading,
  onInputChange,
  onSend,
}: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  return (
    <div className="flex flex-1 flex-col">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white p-4">
        <div className="flex items-center">
          <Bot className="mr-3 h-6 w-6 text-blue-600" />
          <div>
            <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
            <p className="text-sm text-gray-500">
              Web Development AI Assistant
            </p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="max-h-[84.5vh] flex-1 p-4">
        <div className="mx-auto max-w-4xl space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex items-start space-x-3 ${
                msg.role === "user" ? "flex-row-reverse space-x-reverse" : ""
              }`}
            >
              <div
                className={`flex h-8 w-8 items-center justify-center rounded-full ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-200 text-gray-600"
                }`}
              >
                {msg.role === "user" ? (
                  <User className="h-4 w-4" />
                ) : (
                  <Bot className="h-4 w-4" />
                )}
              </div>
              <div
                className={`max-w-3xl rounded-lg px-4 py-3 ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white"
                    : "border border-gray-200 bg-white text-gray-900"
                }`}
              >
                <div className="prose max-w-none break-words">
                  <MarkdownRenderer markdown={msg.content} />
                </div>
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* Input */}
      <form onSubmit={onSend} className="border-t border-gray-200 bg-white p-4">
        <div className="mx-auto flex max-w-4xl space-x-3">
          <Input
            value={input}
            onChange={(e) => onInputChange(e.target.value)}
            placeholder="Ask something..."
            disabled={isLoading}
            className="flex-1"
          />
          <Button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-700"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </form>
    </div>
  );
}
