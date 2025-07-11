"use client";

import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { MessageCircle, Plus, Trash2 } from "lucide-react";
import { ChatInstance } from "@/types/chat";

interface ChatSidebarProps {
  chatInstances: ChatInstance[];
  activeChatId: string | null;
  onSelectChat: (id: string) => void;
  onCreateChat: () => void;
  onDeleteChat: (id: string) => void;
}

export default function ChatSidebar({
  chatInstances,
  activeChatId,
  onSelectChat,
  onCreateChat,
  onDeleteChat,
}: ChatSidebarProps) {
  return (
    <div className="flex w-80 flex-col border-r border-gray-200 bg-white">
      <div className="border-b border-gray-200 p-4">
        <div className="mb-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-800">DevBot AI</h1>
          <Button
            onClick={onCreateChat}
            size="sm"
            className="bg-blue-600 hover:bg-blue-700"
          >
            <Plus className="mr-1 h-4 w-4" />
            New Chat
          </Button>
        </div>
      </div>
      <ScrollArea className="flex-1 p-2">
        <div className="space-y-2">
          {chatInstances?.map((chat) => (
            <Card
              key={chat.id}
              className={`cursor-pointer transition-colors hover:bg-gray-50 ${
                activeChatId === chat.id ? "border-blue-200 bg-blue-50" : ""
              }`}
              onClick={() => onSelectChat(chat.id)}
            >
              <CardContent className="p-3">
                <div className="flex items-start justify-between">
                  <div className="min-w-0 flex-1">
                    <div className="mb-1 flex items-center">
                      <MessageCircle className="mr-2 h-4 w-4 text-gray-400" />
                      <h3 className="truncate text-sm font-medium text-gray-900">
                        {chat.title}
                      </h3>
                    </div>
                    <p className="text-xs text-gray-500">
                      {new Date(chat.createdAt).toLocaleTimeString()}
                    </p>
                    <p className="mt-1 text-xs text-gray-400">
                      {chat.messages.length} messages
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteChat(chat.id);
                    }}
                    className="hover:text-red-600"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
