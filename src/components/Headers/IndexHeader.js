/*eslint-disable*/
import React from "react";

// reactstrap components
import { Container } from "reactstrap";
// import { Avatar } from '@mui/material';
// core components



function IndexHeader() {
  let pageHeader = React.createRef();

  React.useEffect(() => {
    if (window.innerWidth > 991) {
      const updateScroll = () => {
        let windowScrollTop = window.pageYOffset / 3;
        pageHeader.current.style.transform =
          "translate3d(0," + windowScrollTop + "px,0)";
      };
      window.addEventListener("scroll", updateScroll);
      return function cleanup() {
        window.removeEventListener("scroll", updateScroll);
      };
    }
  });

  return (
    <>
      <div className="page-header clear-filter" filter-color="blue">
        <div
          className="page-header-image"
          style={{
            backgroundImage: "url(" + require("assets/img/background1.png") + ")"
          }}
          ref={pageHeader}
        ></div>
        
        <Container>
          <div className="content-center brand">
            <img
              alt="..."
              className="n-logo"
              src={require("assets/img/avatarnew.png")}
            ></img>
            {/* <Avatar
              alt="Yuheng Fan"
              src={require("assets/img/avatar.png")}
              className={classes.avatar}
            /> */}
            <h1 className="h1-seo">Welcome to YUHENG FAN blog</h1>
            <h3>This is my personal blog to share my study notes, interests, music, movies, games, etc </h3>
          </div>

          
          
            
        </Container>
      </div>
    </>
  );
}

export default IndexHeader;
