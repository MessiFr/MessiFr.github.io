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
import data from "views/documents/plugin/doc";

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
  // const [id, setId] = React.useState(0);
  const [curr, setCurr] = React.useState({id: '0', path: 'docs/homepage.md'})

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
    setCurr({id: '0', path: 'docs/homepage.md'});
  };

  // function renderTree(data) {
    
  //   return data.map((item) => (
  //     <TreeItem 
  //       key={item.id} 
  //       nodeId={item.id.toString()} 
  //       label={<IndexFont label={item.name} />} 
  //       onClick={(event) => handleClickId(event, item)}
  //       >
  //       {item.children && renderTree(item.children)}
  //     </TreeItem>
  //   ));
  // }

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

  const handleExpandClick = () => {
    setExpanded((oldExpanded) =>
      oldExpanded.length === 0 ? ["1", "4", "27", "5", "28", "30"] : [],
    );
  };

  console.log(expanded);
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
                <h3 className="title" style={{ textAlign: 'left' }} onClick={(event) => handleClickHomePage(event)}>Documents</h3>
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
                  {renderTree(data)}
                </TreeView>
              </Box>
              
            </Grid>
            <Grid item xs={8} sx={{ p: 2, alignItems: 'flex-start', marginRight: '2%'}}>
              {/* 文章内容 */}
              <DocumentContext item={curr}/>
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
