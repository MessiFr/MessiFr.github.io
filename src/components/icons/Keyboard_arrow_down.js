// import { KeyboardArrowDown } from '@material-ui/icons';
import React from 'react';

function KeyboardArrowDown(props) {
	const fill = props.fill || 'currentColor';
	// const secondaryfill = props.secondaryfill || fill;
	// const strokewidth = props.strokewidth || 100;
	const width = props.width || '10%';
	const height = props.height || '10%';

	return (
		<svg height={height} width={width} viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
			<g fill="none">
				<path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z" fill={fill}/>
			</g>
		</svg>
	);
};

export default KeyboardArrowDown;