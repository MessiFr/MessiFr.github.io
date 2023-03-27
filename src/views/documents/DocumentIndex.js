import * as React from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import TreeView from '@mui/lab/TreeView';
import TreeItem from '@mui/lab/TreeItem';
import { Typography } from '@mui/material';

function IndexFont(props) {
  return (
    <Typography component="div"><Box sx={{ fontWeight: 'bold', m: 1 }}>{props.label}</Box></Typography>
  );
}


export default function DocumentIndex() {
  const [expanded, setExpanded] = React.useState([]);
  const [selected, setSelected] = React.useState([]);

  const handleToggle = (event, nodeIds) => {
    setExpanded(nodeIds);
  };

  const handleSelect = (event, nodeIds) => {
    setSelected(nodeIds);
  };

  const handleExpandClick = () => {
    setExpanded((oldExpanded) =>
      oldExpanded.length === 0 ? ['1', '2'] : [],
    );
  };

  return (
    <Box sx={{ height: 600, flexGrow: 1, maxWidth: 400, overflowY: 'auto' }}>
      <h3 className="title" style={{ textAlign: 'left' }}>Menu</h3>
      <Box sx={{ mb: 1 }}>
        <Button onClick={handleExpandClick} size='small'>
          {expanded.length === 0 ? 'Expand all' : 'Collapse all'}
        </Button>
        {/* <Button onClick={handleSelectClick}>
          {selected.length === 0 ? 'Select all' : 'Unselect all'}
        </Button> */}
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
        <TreeItem nodeId="1" label={<IndexFont label="1. Notes" />}>
          <TreeItem nodeId="3" label={<IndexFont label="1.1 NLP" />} />
          <TreeItem nodeId="4" label={<IndexFont label="1.2 Cluster Cloud Computing" />} />
          <TreeItem nodeId="5" label={<IndexFont label="1.3 Machine Learning" />} />
        </TreeItem>
        <TreeItem nodeId="2" label={<IndexFont label="2. Programs" />}>
          <TreeItem nodeId="6" label={<IndexFont label="2.1 Cluster Cloud Computing" />} />
          <TreeItem nodeId="7" label={<IndexFont label="2.2 Machine Learning" />} />
          <TreeItem nodeId="8" label={<IndexFont label="2.3 NLP" />} />
        </TreeItem>
      </TreeView>
    </Box>
  );
}