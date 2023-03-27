import { Link } from 'react-router-dom';

const documents = [
  { id: 1, title: 'Document 1', content: 'This is the content of Document 1.' },
  { id: 2, title: 'Document 2', content: 'This is the content of Document 2.' },
  { id: 3, title: 'Document 3', content: 'This is the content of Document 3.' },
];

function DocumentList() {
  return (
    <div>
      <h2>Documents</h2>
      <ul>
        {documents.map((document) => (
          <li key={document.id}>
            <Link to={`/documents/${document.id}`}>{document.title}</Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
