//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   DTK_DataSource.hpp
 * \author Stuart Slattery
 * \date   Thu Nov 17 07:53:43 2011
 * \brief  Interface declaration for data source applications.
 */
//---------------------------------------------------------------------------//

#ifndef DTK_DATASOURCE_HPP
#define DTK_DATASOURCE_HPP

#include <string>

#include <Teuchos_Describable.hpp>

namespace DataTransferKit
{

//===========================================================================//
/*!
 * \class DataSource
 * \brief Protocol declaration for applications acting as a data source in
 * multiphysics coupling. 
 *
 * This interface is templated on the type of field data being
 * transferred, the handle type for mesh entities, and the coordinate
 * type. Ordinal type for communication is int. Implementing this interface
 * will then provide a single mesh and the fields associated with that
 * mesh. Multiple meshes will need multiple data source interfaces specified.
 */
/*! 
 * \example core/test/tstInterfaces.cc
 *
 * Test of DataSource.
 */
//===========================================================================//
class DataSource : Teuchos::Describable
{
  public:

    /*!
     * \brief Constructor.
     */
    DataSource()
    { /* ... */ }

    /*!
     * \brief Destructor.
     */
    virtual ~DataSource()
    { /* ... */ }

    /*!
     * \brief Get the communicator object for the physics implementing this
     * interface.
     * \return The communicator for the source application. This class is
     * required to implement MPI primitives and use reference semantics.
     */
    template<class Communicator>
    virtual void getSourceComm( const Communicator &source_comm ) = 0;

    /*!
     * \brief Check whether or not a field is supported. Return false if this
     * field is not supported.
     * \param field_name The name of the field for which support is being
     * checked.
     */
    virtual bool isFieldSupported( const std::string &field_name ) = 0;

    /*! 
     * \brief Provide the local source mesh nodes this includes nodes needed
     * to resolve higher order elements.
     * \param source_nodes View of the local source mesh nodes. This view
     * is required to persist. The NodeField type is expected to implement
     * FieldTraits. NodeField::value_type is expected to implement
     * NodeTraits. 
     */
    template<typename NodeField>
    virtual void getSourceMeshNodes( const NodeField &source_nodes ) = 0;

    /*! 
     * \brief Provide the local source mesh elements.
     * \param source_elements View of the local source mesh
     * elements. This view is required to persist. The ElementField object
     * passed must be a contiguous memory view of DataTransferKit::Element
     * objects. The ElementField type is exepected to implement
     * FieldTraits. ElementField::value_type is expected to implement
     * ElementTraits. 
     */
    template<class ElementField>
    virtual void getSourceMeshElements( const ElementField &source_elements) = 0;

    /*! 
     * \brief Provide a const view of the local source data at the source mesh
     * nodes. 
     * \param field_name The name of the field to provide data from.
     * \param source_node_data A persisting view of data to be sent. There are
     * two requirements for this view: 1) it is of size equal to the number of
     * source mesh nodes in the local domain, 2) the data is in the same order as the
     * nodes found provided by getSourceMeshNodes. This view is required to
     * persist. The DataField type is expected to implement
     * FieldTraits. ElementField::value_type is expected to implement
     * Teuchos::ScalarTraits. 
     */
    template<class DataField>
    virtual void getSourceNodeData( const std::string &field_name,
				    DataField &source_node_data ) = 0;

    /*!
     * \brief Given a field, get a global data element to be sent to the
     * target. 
     * \param field_name The name of the field to get data from.
     * \return The global data element.
     */
    virtual DataType 
    getGlobalSourceData( const std::string &field_name ) = 0;
};

} // end namespace DataTransferKit

#include "DataTransferKit_DataSource_Def.hpp"

#endif // DTK_DATASOURCE_HPP

//---------------------------------------------------------------------------//
// end of DTK_DataSource.hpp
//---------------------------------------------------------------------------//