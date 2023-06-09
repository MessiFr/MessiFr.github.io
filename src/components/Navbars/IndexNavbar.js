import React from "react";
import { useHistory } from "react-router-dom";
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


function IndexNavbar(props) {
  const [navbarColor, setNavbarColor] = React.useState("navbar-transparent");
  const [collapseOpen, setCollapseOpen] = React.useState(false);
  React.useEffect(() => {
    const updateNavbarColor = () => {
      if (
        document.documentElement.scrollTop > 399 ||
        document.body.scrollTop > 399
      ) {
        setNavbarColor("");
      } else if (
        document.documentElement.scrollTop < 400 ||
        document.body.scrollTop < 400
      ) {
        setNavbarColor("navbar-transparent");
      }
    };
    window.addEventListener("scroll", updateNavbarColor);
    return function cleanup() {
      window.removeEventListener("scroll", updateNavbarColor);
    };
  });

  const history = useHistory();

  const handleClick = (pathName) => (e) => {
    e.preventDefault();
    history.push(`${pathName}`);
  };
  

  // const rootUrl = process.env.PUBLIC_URL;

  return (
    <>
      {collapseOpen ? (
        <div
          id="bodyClick"
          onClick={() => {
            document.documentElement.classList.toggle("nav-open");
            setCollapseOpen(false);
          }}
        />
      ) : null}
      <Navbar className={"fixed-top " + navbarColor} expand="lg" color='info'>
        <Container>
          <div className="navbar-translate" style={{position: "absolute", left: "10%"}}>
          
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
                <DropdownItem href="/" >
                  HomePage
                </DropdownItem>
                <DropdownItem onClick={handleClick('/documents')} >
                  Documents
                </DropdownItem>
                <DropdownItem href="#pablo" >
                  Resume
                </DropdownItem>
                <DropdownItem onClick={handleClick('/gallery')} >
                  Gallery
                </DropdownItem>
                <DropdownItem divider></DropdownItem>
                <DropdownItem onClick={handleClick('/profile-page')} >
                  About me
                </DropdownItem>
                <DropdownItem divider></DropdownItem>
                <DropdownItem onClick={handleClick('/profile-page')} >
                  Contacts
                </DropdownItem>
              </DropdownMenu>
            </UncontrolledDropdown>
            
            <button
              className="navbar-toggler navbar-toggler"
              onClick={() => {
                document.documentElement.classList.toggle("nav-open");
                setCollapseOpen(!collapseOpen);
              }}
              aria-expanded={collapseOpen}
              type="button"
            >
              <span className="navbar-toggler-bar top-bar"></span>
              <span className="navbar-toggler-bar middle-bar"></span>
              <span className="navbar-toggler-bar bottom-bar"></span>
            </button>
          </div>
          <Collapse
            className="justify-content-end"
            isOpen={collapseOpen}
            navbar
            style={{ width: "42%" }}
          >
            <Nav navbar style={{position: "absolute", right: "10%"}}>
              {collapseOpen ? (
                <>
                  <NavItem>
                    <NavLink
                      href="/index"
                      target="_blank"
                      id="home-tooltip"
                    >                 
                      <p className="d-lg-none d-xl-none">Home Page</p>
                    </NavLink>
                  </NavItem>
                  <NavItem>
                    <NavLink
                      target="_blank"
                      id="doc-tooltip"
                      onClick={handleClick('/documents')}
                    >                 
                      <p className="d-lg-none d-xl-none">Documents</p>
                    </NavLink>
                  </NavItem>
                  <NavItem>
                    <NavLink
                      onClick={handleClick('/gallery')}
                      target="_blank"
                      id="gallery-tooltip"
                    >                 
                      <p className="d-lg-none d-xl-none">Gallery</p>
                    </NavLink>
                  </NavItem>
                  <NavItem>
                    <NavLink
                      onClick={handleClick('/profile-page')}
                      target="_blank"
                      id="profile-tooltip"
                    >                 
                      <p className="d-lg-none d-xl-none">About Me</p>
                    </NavLink>
                  </NavItem>
                </>
              ):(
                <>
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
                  href="https://www.facebook.com/yuheng.fan.39/friends/"
                  target="_blank"
                  id="Facebook-tooltip"
                >
                  <i className="fab fa-facebook"></i>
                  <p className="d-lg-none d-xl-none">LinkedIn</p>
                </NavLink>
                <UncontrolledTooltip target="#Facebook-tooltip">
                  Like me on Facebook
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
                </>
              )}
              
              
            </Nav>
          </Collapse>
        </Container>
      </Navbar>
    </>
  );
}

export default IndexNavbar;
