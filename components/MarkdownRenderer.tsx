import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";

export default function MarkdownRenderer({ markdown }: { markdown: string }) {
  return (
    <div className="prose max-w-none break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          code(props) {
            const { className, children } = props;
            const isInline = !className;

            return isInline ? (
              <code className="rounded bg-gray-100 px-1 py-0.5 text-sm break-words whitespace-pre-wrap">
                {children}
              </code>
            ) : (
              <pre className="rounded bg-gray-100 p-4 text-sm break-words whitespace-pre-wrap">
                <code className={className}>{children}</code>
              </pre>
            );
          },
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
