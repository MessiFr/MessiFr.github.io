import React from "react";

// reactstrap components
// import {
//   Container,
// } from "reactstrap";

import SolidNavbar from "components/Navbars/SolidNavbar";
import DefaultFooter from "components/Footers/DefaultFooter.js";

import DocumentIndex from "views/documents/DocumentIndex";

import { Grid } from '@mui/material';
// import DocumentContext from "views/documents/DocumentContext";

function DocumentPage() {
  
  return (
    <>
      <SolidNavbar label="Documents"/>
      <div className="wrapper">
        <div className="section">
          <Grid container spacing={0} sx={{height: '180vh'}}>
            <Grid item xs={2.7} sx={{ p: 2, alignItems: 'flex-start'}} style={{ marginLeft: "1%" }}>
              {/* 左边的内容 */}
              <DocumentIndex/>
            </Grid>
            <Grid item xs={12} sx={{ background:"grey.100", p: 2}}>
              {/* 右边的内容 */}
              {/* <DocumentContext/> */}
            </Grid>
          </Grid>
        </div>
        <DefaultFooter />
      </div>
    </>
  );
}

export default DocumentPage;
