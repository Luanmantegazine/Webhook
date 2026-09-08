import ReactPaginate from "react-paginate";
import React from "react";

const Pagination = ({ pageCount, onPageChange }) => (
  <ReactPaginate
    nextLabel=">"
    previousLabel="<"
    breakLabel="..."
    pageRangeDisplayed={3}
    marginPagesDisplayed={1}
    onPageChange={onPageChange}
    pageCount={pageCount}
    containerClassName="flex gap-2 justify-center items-center py-4"
    pageClassName="px-3 py-1 rounded-lg border text-sm"
    activeClassName="bg-sky-600 text-white"
  />
);

export default Pagination;