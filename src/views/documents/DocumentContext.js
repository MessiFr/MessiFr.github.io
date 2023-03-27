import { Viewer } from '@react-pdf-viewer/core';
import { Box } from '@mui/system';
import { Worker } from '@react-pdf-viewer/core';


export default function DocumentContext() {
    return (
        <>
        <Box sx={{ height: 600, flexGrow: 1, maxWidth: 400, overflowY: 'auto' }}>
            <div
                style={{
                    border: '1px solid rgba(0, 0, 0, 0.3)',
                    height: '750px',
                }}
            >
            <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.4.120/build/pdf.worker.min.js">
                <Viewer fileUrl="../ccc.pdf" options={{workerSrc: "pdf.worker.js"}}/>
            </Worker>                
            </div>
        </Box>
            
        </>
    )
}