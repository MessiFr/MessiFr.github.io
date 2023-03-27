import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import fileInfo from "./doc";
import { Container } from "reactstrap";

export default function DocumentContext(props)  {
  const [content, setContent] = useState("");

  useEffect(() => {
    
    const path = `/docs/${fileInfo[props.id][0]}`; // "./docs/notes/nlp/assignment_1"
    console.log(path)

    fetch(path)
      .then((res) => res.text())
      .then((text) => setContent(text));
  }, [props.id]);

  return (
    <Container>
    <h3 className="title" style={{ textAlign: 'center' }}>{fileInfo[props.id][1]}</h3>
    <div className="post">
      <ReactMarkdown children={content} />
    </div>
    </Container>
    
  );
};
