import React from "react";
import { useHistory } from "react-router-dom";
// import Typography from '@mui/material/Typography';
// import { withStyles } from "@material-ui/core/styles";
// import qrcode from 'qrcode';
// reactstrap components
import {
  // Button,
  Collapse,
  DropdownToggle,
  DropdownMenu,
  DropdownItem,
  UncontrolledDropdown,
  // NavbarBrand,
  Navbar,
  NavItem,
  NavLink,
  Nav,
  Container,
  UncontrolledTooltip,
} from "reactstrap";

function SolidNavbar(props) {
  const [collapseOpen, setCollapseOpen] = React.useState(false);
  
  // const WhiteTextTypography = withStyles({
  //   root: {
  //     color: "#FFFFFF"
  //   }
  // })(Typography);

  const history = useHistory();

  const handleProfileClick = (e) => {
    e.preventDefault();
    history.push('/profile-page');
  };

  const handleHomePageClick = (e) => {
    e.preventDefault();
    history.push('/index');
  };


  const handleDocumentPageClick = (e) => {
    e.preventDefault();
    history.push('/documents');
  };

  return (
    <>
      <style>
        {`@import url('https://fonts.cdnfonts.com/css/midpoint-pro');`}
      </style>
      {collapseOpen ? (
        <div
          id="bodyClick"
          onClick={() => {
            document.documentElement.classList.toggle("nav-open");
            setCollapseOpen(false);
          }}
        />
      ) : null}
      <Navbar className={"fixed-top"} expand="lg" style={{ backgroundColor: "black" }} >
        <Container style={{ marginLeft: 0, marginRight:0 }}>
          <div className="navbar-translate" style={{ marginLeft: "0%" }}>
          <UncontrolledDropdown className="button-dropdown">
            <DropdownToggle
              caret
              data-toggle="dropdown"
              href="#pablo"
              id="navbarDropdown"
              tag="a"
              onClick={(e) => e.preventDefault()}
            >
              <span className="button-bar"></span>
              <span className="button-bar"></span>
              <span className="button-bar"></span>
            </DropdownToggle>
            <DropdownMenu aria-labelledby="navbarDropdown">
              {/* <DropdownItem header tag="a">
                Dropdown header
              </DropdownItem> */}
              <DropdownItem href="/" onClick={handleHomePageClick}>
                HomePage
              </DropdownItem>
              <DropdownItem href="/documents" onClick={handleDocumentPageClick}>
                Documents
              </DropdownItem>
              <DropdownItem href="#pablo" onClick={(e) => e.preventDefault()}>
                Resume
              </DropdownItem>
              <DropdownItem divider></DropdownItem>
              <DropdownItem href="/profile-page" onClick={handleProfileClick}>
                About me
              </DropdownItem>
              <DropdownItem divider></DropdownItem>
              <DropdownItem href="/profile-page" onClick={handleProfileClick}>
                Contacts
              </DropdownItem>
            </DropdownMenu>
          </UncontrolledDropdown>   
          </div>
          
          <Collapse
            className="justify-content-end"
            isOpen={collapseOpen}
            navbar
            style={{ marginRight: "-20%" }}
          >
            <Nav navbar>
              <NavItem>
                <NavLink
                  href="https://github.com/MessiFr"
                  target="_blank"
                  id="github-tooltip"
                >
                  <i className="fab fa-github"></i>
                  <p className="d-lg-none d-xl-none">Github</p>
                </NavLink>
                <UncontrolledTooltip target="#github-tooltip">
                  Follow me on Github
                </UncontrolledTooltip>
              </NavItem>
              <NavItem>
                <NavLink
                  href="https://www.linkedin.com/in/yuheng-fan-b915917b/"
                  target="_blank"
                  id="LinkedIn-tooltip"
                >
                  <i className="fab fa-linkedin-in"></i>
                  <p className="d-lg-none d-xl-none">LinkedIn</p>
                </NavLink>
                <UncontrolledTooltip target="#LinkedIn-tooltip">
                  Like me on LinkedIn
                </UncontrolledTooltip>
              </NavItem>
              <NavItem>
                <NavLink
                  href="https://www.instagram.com/yuhengfan"
                  target="_blank"
                  id="instagram-tooltip"
                >
                  <i className="fab fa-instagram"></i>
                  <p className="d-lg-none d-xl-none">Instagram</p>
                </NavLink>
                <UncontrolledTooltip target="#instagram-tooltip">
                  Follow me on Instagram
                </UncontrolledTooltip>
              </NavItem>
              <NavItem>
                <NavLink
                  // href="https://www.linkedin.com/in/yuheng-fan-b915917b/"
                  target="_blank"
                  id="Wechat-tooltip"
                >
                  <i className="fab fa-weixin"></i>
                  <p className="d-lg-none d-xl-none">WeChat</p>
                </NavLink>
                <UncontrolledTooltip target="#Wechat-tooltip">
                  <img src={require("assets/img/qrcode.jpeg")} alt="QR code" style={{ width: '200px', height: '200px' }} />
                </UncontrolledTooltip>
              </NavItem>
            </Nav>
          </Collapse>
        </Container>
      </Navbar>
    </>
  );
}

export default SolidNavbar;