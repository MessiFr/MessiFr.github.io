import { Worker } from '@react-pdf-viewer/core';

// Import the main component
import { Viewer, SpecialZoomLevel } from '@react-pdf-viewer/core';

// Import the styles
import '@react-pdf-viewer/core/lib/styles/index.css';
import { bookmarkPlugin } from '@react-pdf-viewer/bookmark';


export default function DocumentPdfContext(props) {

	// Create new plugin instance
	const bookmarkPluginInstance = bookmarkPlugin();
	// console.log(SpecialZoomLevel);

	return (
		<div
			style={{
				border: '1px solid rgba(0, 0, 0, 0.3)',
				height: '750px',
			}}
		>
			<Worker workerUrl="https://unpkg.com/pdfjs-dist@3.4.120/build/pdf.worker.min.js">
				<Viewer fileUrl={`/${props.item.path}`} plugins={[bookmarkPluginInstance]} defaultScale={SpecialZoomLevel.PageWidth}/>
			</Worker>
		</div>
	)
}

// export default function DocumentPdfContext() {
// 	return (
// 		<div></div>
// 	)
// }