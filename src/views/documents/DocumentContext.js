import { useEffect, useState } from "react";

import Zoom from "@material-ui/core/Zoom";
import PropTypes from "prop-types";
import useScrollTrigger from "@material-ui/core/useScrollTrigger";
import { makeStyles } from "@material-ui/core/styles";

// import rehypeKatex from 'rehype-katex'
import rehypeRaw from "rehype-raw";
import ReactMarkdown from 'react-markdown';
// import remarkMath from 'remark-math';
import remarkHtml from 'remark-html';
import remarkGfm from 'remark-gfm';
import SyntaxHighlighter from "react-syntax-highlighter/dist/esm/default-highlight";
import Fab from "@material-ui/core/Fab";
import KeyboardArrowUpIcon from '@material-ui/icons/KeyboardArrowUp';

import { MathJaxContext, MathJax } from 'better-react-mathjax';
import { Container } from "reactstrap";



const useStyles = makeStyles(theme => ({
  root: {
    position: "fixed",
    bottom: theme.spacing(2),
    right: theme.spacing(2)
  }
}));

function ScrollTop(props) {
  const { children } = props;
  const classes = useStyles();
  const trigger = useScrollTrigger({
    disableHysteresis: true,
    threshold: 100
  });

  const handleClick = event => {
    const anchor = (event.target.ownerDocument || document).querySelector(
      "#back-to-top-anchor"
    );

    if (anchor) {
      anchor.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  return (
    <Zoom in={trigger}>
      <div onClick={handleClick} role="presentation" className={classes.root}>
        {children}
      </div>
    </Zoom>
  );
}

ScrollTop.propTypes = {
  children: PropTypes.element.isRequired
};

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
    // console.log(props)
    const path = `/${props.item.path}`; // "./docs/notes/nlp/assignment_1"
    console.log(path)

    fetch(path)
      .then((res) => res.text())
      .then((text) => setContent(text));
  }, [props]);


  return (
    <Container>
    {/* <h3 className="title" style={{ textAlign: 'center' }}>{props.item.title}</h3> */}
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
    <ScrollTop>
      <Fab color="primary" size="medium" aria-label="scroll back to top">
        <KeyboardArrowUpIcon />
      </Fab>
    </ScrollTop>
    </Container>
    
  );
};
