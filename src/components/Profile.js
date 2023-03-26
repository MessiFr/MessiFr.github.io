import React from 'react';
import { makeStyles } from '@mui/styles';
import { Avatar, Typography, Grid, IconButton } from '@mui/material';
import { GitHub, Instagram, LinkedIn } from '@mui/icons-material';

const useStyles = makeStyles((theme) => ({
  root: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: theme.spacing(2),
  },
  avatar: {
    width: theme.spacing(20),
    height: theme.spacing(20),
    margin: theme.spacing(1),
  },
  name: {
    fontWeight: 'bold',
  },
  school: {
    margin: theme.spacing(1),
  },
  icon: {
    margin: theme.spacing(1),
  },
}));

function Profile() {
  const classes = useStyles();

  return (
    <Grid container className={classes.root}>
      <Avatar
        alt="Yuheng Fan"
        src="/static/icons/avatar.jpeg" 
        className={classes.avatar}
      />
      <Typography variant="h4" className={classes.name}>
        Yuheng Fan
      </Typography>
      <Typography variant="h10" className={classes.school}>
        The University of Melbourne
      </Typography>
      <Grid item>
        <IconButton
          href="https://github.com/MessiFr"
          target="_blank"
          className={classes.icon}
        >
          <GitHub />
        </IconButton>
        <IconButton
          href="https://www.linkedin.com/in/yuheng-fan-b915917b/"
          target="_blank"
          className={classes.icon}
        >
          <LinkedIn />
        </IconButton>
        <IconButton
          href="https://instagram.com/yuhengfan"
          target="_blank"
          className={classes.icon}
        >
          <Instagram />
        </IconButton>
      </Grid>
    </Grid>
  );
}

export default Profile;
