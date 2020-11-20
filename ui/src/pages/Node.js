import React from "react";
import cn from "classnames";

import { tilePixelVector, getNodeDelta } from "../utils/coordinates";

function Building({ building }) {
  const color = `bg-white bg-${building[0].toLowerCase()}-700`;
  const city = building[1] === "CITY";
  const border = city ? "w-8 h-8 border-2" : "w-6 h-6 border-2";
  return (
    <>
      <div
        className={cn("node-building absolute border-black", color, border)}
      ></div>
      {city && (
        <div
          className={cn(
            "node-building absolute border-black",
            color,
            "w-4 h-4 border-2"
          )}
          style={{ transform: "translateY(0.50rem) translateX(0.50rem)" }}
        ></div>
      )}
    </>
  );
}

export default function Node({
  id,
  centerX,
  centerY,
  w,
  h,
  size,
  coordinate,
  direction,
  building,
}) {
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const [deltaX, deltaY] = getNodeDelta(direction, w, h);
  const x = tileX + deltaX;
  const y = tileY + deltaY;

  return (
    <div
      className="node w-8 h-8 absolute"
      style={{
        left: x,
        top: y,
        transform: `translateY(-0.75rem) translateX(-0.75rem)`,
      }}
      onClick={() => console.log("Clicked node", id)}
    >
      {building && <Building building={building} />}
    </div>
  );
}
