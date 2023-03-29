import React from "react";
import { useEffect, useState } from "react";
import { useHistory } from 'react-router-dom';
import SolidNavbar from "components/Navbars/SolidNavbar";
import DefaultFooter from "components/Footers/DefaultFooter.js";
import DocumentContext from "views/documents/DocumentContext";
// import DocumentPdfContext from "views/documents/DocuemntPdfContext";
import Box from '@mui/material/Box';
import Toolbar from "@material-ui/core/Toolbar";
import Button from '@mui/material/Button';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import { styled } from '@mui/material/styles';
import TreeView from '@mui/lab/TreeView';
import TreeItem from '@mui/lab/TreeItem';
import data from "views/documents/info/doc";

import { Grid } from '@mui/material';
import { Typography } from '@mui/material';


function IndexFont(props) {
  return (
    <Typography component="div"><Box sx={{ fontWeight: 'bold', m: 1 }}>{props.label}</Box></Typography>
  );
}


function DocumentPage() {
  const homepageConfig = {id: '0', path: 'docs/homepage.md', doc_type: 'md'};

  const [expanded, setExpanded] = useState([]);
  const [selected, setSelected] = useState([]);
  
  const [expandId, setExpandId] = useState([]);
  const [curr, setCurr] = useState(homepageConfig);

  // const { label, nodeId } = props;
  const history = useHistory();

  const handleClickId = (event, item) => {
    // console.log(item)
    if (item.path) {
      history.push(`/documents/${item.id}`);
      setCurr(item);
    }
  };


  const handleClickHomePage = (event) => {
    
    history.push(`/documents/`);
    setCurr(homepageConfig);
  };


  function renderTree(data, parentIndex = '') {
    return data.map((item, index) => {
      const currentIndex = parentIndex ? `${parentIndex}.${index + 1}` : `${index + 1}`;
  
      return (
        <TreeItem 
          key={item.id} 
          nodeId={item.id.toString()} 
          label={<IndexFont label={`${currentIndex}. ${item.name}`} />} 
          onClick={(event) => handleClickId(event, item)}
        >
          {item.children && renderTree(item.children, currentIndex)}
        </TreeItem>
      )
    });
  }

  const handleToggle = (event, nodeIds) => {
    setExpanded(nodeIds);
  };

  const handleSelect = (event, nodeIds) => {
    setSelected(nodeIds);
  };

  useEffect(() => {
    const getAllIds = (data) => {
      let ids = [];
      for (let i = 0; i < data.length; i++) {
        ids.push(data[i].id.toString());
        if (data[i].children) {
          ids = [...ids, ...getAllIds(data[i].children)];
        }
      }
      return ids;
    };
    
    setExpandId(getAllIds(data));
    
  }, []);
  
  const handleExpandClick = () => {
    console.log(expandId);
    setExpanded((oldExpanded) =>
      oldExpanded.length === 0 ? expandId : [],
    );
  };

  const StyledGrid = styled(Grid)({
    borderRight: '2px solid rgba(0, 0, 0, 0.1)',
    position: 'relative',
  });

  return (
    <>
      <SolidNavbar label="Documents"/>
      <div className="wrapper">
        <div className="section">

          <Toolbar id="back-to-top-anchor" />
          <div style={{ marginTop: '-5%' }}> 
            <Grid container spacing={2} >
              <StyledGrid item xs={2.8} sx={{ p: 2, alignItems: 'flex-start', marginLeft: '2%'}}>

                {/* 目录 */}
                <Box 
                  sx={{ 
                    height: 1000, 
                    flexGrow: 1, 
                    maxWidth: 400, 
                    overflowY: 'auto'
                  }}
                  >
                    
                  <h3 className="title" style={{ textAlign: 'left' }} onClick={(event) => handleClickHomePage(event)}>Documents</h3>

                  <Box sx={{ mb: 1 }}>
                    <Button variant='outlined' onClick={handleExpandClick} size='small'>
                      {expanded.length === 0 ? 'Expand all' : 'Collapse all'}
                    </Button>
                  </Box>

                  <TreeView
                    aria-label="controlled"
                    defaultCollapseIcon={<ExpandMoreIcon />}
                    defaultExpandIcon={<ChevronRightIcon />}
                    expanded={expanded}
                    selected={selected}
                    onNodeToggle={handleToggle}
                    onNodeSelect={handleSelect}
                    multiSelect
                  >
                    {renderTree(data)}
                  </TreeView>
                </Box>
                
              </StyledGrid>
              
              <Grid item xs={8} sx={{ p: 2, alignItems: 'flex-start', marginRight: '2%'}}>
                <Box sx={{ height: 1000, flexGrow: 1, overflowY: 'auto' }}>
                  {curr.doc_type === 'md' ? <DocumentContext item={curr}/> : null}
                  {/* {curr.doc_type === 'pdf' ? <DocumentPdfContext item={curr}/> : null} */}
                </Box>
              </Grid>
            </Grid>
          </div>
        </div>
        <div style={{ marginBottom: '0px' }}>
        <DefaultFooter />
        </div>
       
      </div>
    </>
  );
}

export default DocumentPage;
