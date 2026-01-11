import React from "react";
import { Link, useHistory } from "react-router-dom";

// reactstrap components
import {
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

function ExamplesNavbar() {
  const [navbarColor, setNavbarColor] = React.useState("navbar-transparent");
  const [collapseOpen, setCollapseOpen] = React.useState(false);
  
  React.useEffect(() => {
    let timeoutId = null;
    
    const updateNavbarColor = () => {
      // 防止在组件卸载后仍然执行
      if (typeof document !== 'undefined') {
        const scrolled = (document.documentElement && document.documentElement.scrollTop) || 
                         (document.body && document.body.scrollTop);
        
        if (scrolled > 399) {
          setNavbarColor("");
        } else if (scrolled <= 400) {
          setNavbarColor("navbar-transparent");
        }
      }
    };
    
    window.addEventListener("scroll", updateNavbarColor);
    
    return function cleanup() {
      window.removeEventListener("scroll", updateNavbarColor);
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, []);
  
  const history = useHistory();

  const handleProfileClick = (e) => {
    e.preventDefault();
    history.push('/profile-page');
  };

  const handleHomePageClick = (e) => {
    e.preventDefault();
    history.push('/index');
  };

  const handleTradesPageClick = (e) => {
    e.preventDefault();
    history.push('/trades');
  };

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
      <Navbar className={"fixed-top " + navbarColor} color="info" expand="lg">
        <Container>
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
              <DropdownItem href="#pablo" onClick={handleHomePageClick}>
                HomePage
              </DropdownItem>
              <DropdownItem href="#pablo" onClick={handleTradesPageClick}>
                Trades
              </DropdownItem>
              <DropdownItem href="#pablo" onClick={(e) => e.preventDefault()}>
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
              <DropdownItem href="#pablo" onClick={(e) => e.preventDefault()}>
                Contacts
              </DropdownItem>
            </DropdownMenu>
          </UncontrolledDropdown>
          <div className="navbar-translate">           
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
          >
            <Nav navbar>
              <NavItem>
                <NavLink to="/index" tag={Link}>
                  Back to Kit
                </NavLink>
              </NavItem>
              <NavItem>
                <NavLink to="/trades" tag={Link}>
                  Trades
                </NavLink>
              </NavItem>
              <NavItem>
                <NavLink href="https://github.com/creativetimofficial/now-ui-kit-react/issues?ref=creativetim">
                  Have an issue?
                </NavLink>
              </NavItem>
              <NavItem>
                <NavLink
                  href="https://twitter.com/CreativeTim?ref=creativetim"
                  target="_blank"
                  id="twitter-tooltip"
                >
                  <i className="fab fa-twitter"></i>
                  <p className="d-lg-none d-xl-none">Twitter</p>
                </NavLink>
                <UncontrolledTooltip target="#twitter-tooltip">
                  Follow us on Twitter
                </UncontrolledTooltip>
              </NavItem>
              <NavItem>
                <NavLink
                  href="https://www.facebook.com/CreativeTim?ref=creativetim"
                  target="_blank"
                  id="facebook-tooltip"
                >
                  <i className="fab fa-facebook-square"></i>
                  <p className="d-lg-none d-xl-none">Facebook</p>
                </NavLink>
                <UncontrolledTooltip target="#facebook-tooltip">
                  Like us on Facebook
                </UncontrolledTooltip>
              </NavItem>
              <NavItem>
                <NavLink
                  href="https://www.instagram.com/CreativeTimOfficial?ref=creativetim"
                  target="_blank"
                  id="instagram-tooltip"
                >
                  <i className="fab fa-instagram"></i>
                  <p className="d-lg-none d-xl-none">Instagram</p>
                </NavLink>
                <UncontrolledTooltip target="#instagram-tooltip">
                  Follow us on Instagram
                </UncontrolledTooltip>
              </NavItem>
            </Nav>
          </Collapse>
        </Container>
      </Navbar>
    </>
  );
}

export default ExamplesNavbar;