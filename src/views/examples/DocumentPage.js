import React from "react";
import { useHistory } from 'react-router-dom';
import SolidNavbar from "components/Navbars/SolidNavbar";
import DefaultFooter from "components/Footers/DefaultFooter.js";
import DocumentContext from "views/documents/DocumentContext";
import Box from '@mui/material/Box';
import Toolbar from "@material-ui/core/Toolbar";
import Button from '@mui/material/Button';
import { makeStyles } from "@material-ui/core/styles";
import useScrollTrigger from "@material-ui/core/useScrollTrigger";
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import TreeView from '@mui/lab/TreeView';
import TreeItem from '@mui/lab/TreeItem';
import Zoom from "@material-ui/core/Zoom";
import PropTypes from "prop-types";
import Fab from "@material-ui/core/Fab";
import KeyboardArrowUpIcon from '@material-ui/icons/KeyboardArrowUp';

import { Grid } from '@mui/material';
import { Typography } from '@mui/material';


function IndexFont(props) {
  return (
    <Typography component="div"><Box sx={{ fontWeight: 'bold', m: 1 }}>{props.label}</Box></Typography>
  );
}

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

function DocumentPage() {
  const [expanded, setExpanded] = React.useState([]);
  const [selected, setSelected] = React.useState([]);
  const [id, setId] = React.useState(0);

  // const { label, nodeId } = props;
  const history = useHistory();

  const handleClick = (event, id) => {
    history.push(`/documents/${id}`);
    setId(id);
  };

  const handleToggle = (event, nodeIds) => {
    setExpanded(nodeIds);
  };

  const handleSelect = (event, nodeIds) => {
    setSelected(nodeIds);
  };

  const handleExpandClick = () => {
    setExpanded((oldExpanded) =>
      oldExpanded.length === 0 ? ['100', '200', '1', '106', '102', '900'] : [],
    );
  };

  return (
    <>
      <SolidNavbar label="Documents"/>
      <div className="wrapper">
        <div className="section">

          <Toolbar id="back-to-top-anchor" />
          <Grid container spacing={2} >
            <Grid item xs={2.8} sx={{ p: 2, alignItems: 'flex-start', marginLeft: '2%'}}>

              {/* 目录 */}
              <Box sx={{ height: 600, flexGrow: 1, maxWidth: 400, overflowY: 'auto' }}>
                <h3 className="title" style={{ textAlign: 'left' }} onClick={(event) => handleClick(event, 0)}>Documents</h3>
                <Box sx={{ mb: 1 }}>
                  <Button onClick={handleExpandClick} size='small'>
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
                  <TreeItem nodeId="100" label={<IndexFont label="1. Notes" />}>
                    <TreeItem nodeId="101" label={<IndexFont label="1.1 NLP" />} > 
                      {/* <TreeItem nodeId="104" label={<IndexFont label="1.1.1 Assignment 1" />} onClick={(event) => handleClick(event, 104)} /> */}
                    </TreeItem>
                    <TreeItem nodeId="106" label={<IndexFont label="1.2 Machine Learning" />} > 
                      <TreeItem nodeId="107" label={<IndexFont label="1.2.1 Linear Regression" />} onClick={(event) => handleClick(event, 107)} />
                      <TreeItem nodeId="108" label={<IndexFont label="1.2.2 Bayes Inference" />} onClick={(event) => handleClick(event, 108)} />
                      <TreeItem nodeId="109" label={<IndexFont label="1.2.3 Gradient Based Training" />} onClick={(event) => handleClick(event, 109)} />
                      <TreeItem nodeId="110" label={<IndexFont label="1.2.4 SVM" />} onClick={(event) => handleClick(event, 110)} />
                      <TreeItem nodeId="111" label={<IndexFont label="1.2.5 Pytorch" />} onClick={(event) => handleClick(event, 111)} />

                    </TreeItem>
                    <TreeItem nodeId="102" label={<IndexFont label="1.3 Cluster Cloud Computing" />} >
                      <TreeItem nodeId="105" label={<IndexFont label="1.3.1 Slides 1" />} onClick={(event) => handleClick(event, 105)} />
                    </TreeItem>
                  </TreeItem>
                  <TreeItem nodeId="200" label={<IndexFont label="2. Programs" />}>
                    <TreeItem nodeId="201" label={<IndexFont label="2.1 Cluster Cloud Computing" />} />
                    <TreeItem nodeId="202" label={<IndexFont label="2.2 Machine Learning" />} />
                    <TreeItem nodeId="203" label={<IndexFont label="2.3 NLP" />} />
                  </TreeItem>
                  <TreeItem nodeId="1" label={<IndexFont label="3. Resume" />}>
                    <TreeItem nodeId='2' label={<IndexFont label="3.1 Resume(zh)" />}/>
                    <TreeItem nodeId='3' label={<IndexFont label="3.2 Resume(en)" />}/>
                  </TreeItem>   
                  <TreeItem nodeId="900" label={<IndexFont label="4. Development"/>}>
                    <TreeItem nodeId="901" label={<IndexFont label=" -test-" />} onClick={(event) => handleClick(event, 901)} />
                  </TreeItem>
                   
                </TreeView>
              </Box>
              
            </Grid>
            <Grid item xs={8} sx={{ p: 2, alignItems: 'flex-start', marginRight: '2%'}}>
              {/* 文章内容 */}
              <DocumentContext id={id}/>
              <ScrollTop>
                <Fab color="primary" size="medium" aria-label="scroll back to top">
                  <KeyboardArrowUpIcon />
                </Fab>
              </ScrollTop>
            </Grid>
          </Grid>
              
        </div>
        <div style={{ marginBottom: '0px' }}>
        <DefaultFooter />
        </div>
       
      </div>
    </>
  );
}

export default DocumentPage;
