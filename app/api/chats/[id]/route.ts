import { NextRequest, NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";
import prisma from "@/lib/prisma";

export async function PATCH(
  req: NextRequest,
  { params }: { params: { id: string } },
) {
  const threadId = params.id;
  const { messages } = await req.json();

  if (!messages || !Array.isArray(messages)) {
    return NextResponse.json({ error: "Invalid messages" }, { status: 400 });
  }

  const createdMessages = await prisma.message.createMany({
    data: messages.map((m: any) => ({
      id: m.id ?? uuidv4(),
      role: m.role,
      content: m.content,
      threadId,
    })),
  });

  return NextResponse.json({
    success: true,
    createdCount: createdMessages.count,
  });
}

// Delete a thread and its messages
export async function DELETE(
  req: NextRequest,
  { params }: { params: { id: string } },
) {
  const threadId = params.id;

  await prisma.chatThread.delete({
    where: { id: threadId },
  });

  return NextResponse.json({ success: true });
}
