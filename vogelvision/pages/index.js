import React, { Fragment } from 'react'
import ImageUpload from './ImageUpload'

export default function Home() {
  return (
    <Fragment>
    <nav className="navbar navbar-dark bg-dark">
        <div className="container">
            <a className="navbar-brand" href="#">Bird Species Identification</a>
            <button className="btn btn-outline-secondary my-2 my-sm-0" type="submit">Help</button>
        </div>
    </nav>
    <div className="container">
        <div id="content" style={{marginTop: 2 + "em"}}>
          <ImageUpload />
        </div>
    </div>
      </Fragment>
  )
}
