import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";

// GET: Get all chat threads with messages
export async function GET() {
  const chats = await prisma.chatThread.findMany({
    include: { messages: true },
    orderBy: { createdAt: "desc" },
  });

  return NextResponse.json(chats);
}

// POST: Create a new chat thread
export async function POST(req: Request) {
  const { title } = await req.json();

  const chat = await prisma.chatThread.create({
    data: { title },
  });

  return NextResponse.json(chat);
}
