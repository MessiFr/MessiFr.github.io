import React from 'react';
import QRCode from 'qrcode.react';
import './InstagramQRCode.css';

const InstagramQRCode = ({ logoURL }) => {
  // const value = "MessiFr";

  const qrStyle = {
    width: 256,
    height: 256,
    background: 'linear-gradient(45deg, #fd1d1d, #833ab4, #fcb045)',
    color: 'transparent'
  };

  return (
    <div >
      <div className="qr-container">
        <div className="qr-gradient-bg">
          <QRCode
            value={"https://messifr.github.io/"}
            size={200}
            bgColor="transparent"
            fgColor= "white"
            style={qrStyle}
            renderAs="svg"
            imageSettings={{
              src: logoURL,
              x: null,
              y: null,
              height: 50,
              width: 50,
              excavate: true,
            }}
          />
        </div>

        {/* <div className="qr-context-container">
          
          {value}
        </div> */}
      </div>
    </div>
  );
};

export default InstagramQRCode;
