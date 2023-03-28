import React from 'react'
import { useEffect, useState } from "react";

import { Container } from "reactstrap";
import ReactMarkdown from 'react-markdown';
import remarkHtml from 'remark-html';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw'
import SyntaxHighlighter from 'react-syntax-highlighter/dist/esm/default-highlight';

// import remarkMath from 'remark-math';
// import rehypeKatex from 'rehype-katex';

import { MathJaxContext, MathJax } from 'better-react-mathjax';


function Test()  {
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
        
      };
  
    useEffect(() => {
      
      const path = `/docs/development/test.md`;   
      fetch(path)
        .then((res) => res.text())
        .then((text) => setContent(text));
    }, []);

    console.log(content);
  
  
    return (
      <Container>
      <h3 className="title" style={{ textAlign: 'center' }}>Test</h3>
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
  

export default class Demo extends React.Component {

  render() {
    return (     
        <Test />
    )
  }
}


