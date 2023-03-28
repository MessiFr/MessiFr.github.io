import { useEffect, useState } from "react";


// import rehypeKatex from 'rehype-katex'
import rehypeRaw from "rehype-raw";
import ReactMarkdown from 'react-markdown';
// import remarkMath from 'remark-math';
import remarkHtml from 'remark-html';
import remarkGfm from 'remark-gfm';
import SyntaxHighlighter from "react-syntax-highlighter/dist/esm/default-highlight";

import { MathJaxContext, MathJax } from 'better-react-mathjax';

// import { BlockMath, InlineMath } from "react-katex";
// import CodeBlock from "./plugin/CodeBlock";
// import './CodeBlock.css';

import fileInfo from "./plugin/doc";
import { Container } from "reactstrap";

export default function DocumentContext(props)  {
  const [content, setContent] = useState("");

  const config = {
    tex2jax: {
      inlineMath: [["$", "$"]],
      displayMath: [["$$", "$$"]]
    }
  };

  const renders = {
    code: ({ node, inline, className, children, ...props }) => {
        const match = /language-(\w+)/.exec(className || "");
        return !inline && match ? (
          <SyntaxHighlighter
            children={String(children).replace(/\n$/, "")}
            language={match[1]}
            {...props}
          />
        ) : (
          <code className={className} {...props}>
            {children}
          </code>
        );
      },
    // math: (opts) => <BlockMath math={opts.value} />,
    // inlineMath: (opts) => <InlineMath math={opts.value} />,
    
  };

  useEffect(() => {
    
    const path = `/docs/${fileInfo[props.id][0]}`; // "./docs/notes/nlp/assignment_1"
    // console.log(path)

    fetch(path)
      .then((res) => res.text())
      .then((text) => setContent(text));
  }, [props.id]);


  return (
    <Container>
    <h3 className="title" style={{ textAlign: 'center' }}>{fileInfo[props.id][1]}</h3>
    <div className="post">
        <MathJaxContext config={config} >
            <MathJax>
            <ReactMarkdown
                remarkPlugins={[remarkHtml, remarkGfm]}
                rehypePlugins={[rehypeRaw]}
                children={content}
                components={renders}
            />
            </MathJax>
        </MathJaxContext>
    </div>
    </Container>
    
  );
};
