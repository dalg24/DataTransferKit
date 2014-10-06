//---------------------------------------------------------------------------//
/*
  Copyright (c) 2014, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the Oak Ridge National Laboratory nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//---------------------------------------------------------------------------//
/*!
 * \brief DTK_GeometricEntity.cpp
 * \author Stuart R. Slattery
 * \brief Geometric entity interface.
 */
//---------------------------------------------------------------------------//

#include "DTK_GeometricEntity.hpp"
#include "DTK_DBC.hpp"

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
// Constructor.
GeometricEntity::GeometricEntity()
{ /* ... */ }

//---------------------------------------------------------------------------//
//brief Destructor.
GeometricEntity::~GeometricEntity()
{ /* ... */ }

//---------------------------------------------------------------------------//
// Return a string indicating the derived entity type.
std::string GeometricEntity::entityType() const
{
    return b_entity_impl->entityType();
}

//---------------------------------------------------------------------------//
// Get the unique global identifier for the entity.
EntityId GeometricEntity::id() const
{ 
    return b_entity_impl->id();
}
    
//---------------------------------------------------------------------------//
// Get the parallel rank that owns the entity.
int GeometricEntity::ownerRank() const
{ 
    return b_entity_impl->ownerRank();
}
//---------------------------------------------------------------------------//
// Return the physical dimension of the entity.
int GeometricEntity::physicalDimension() const
{ 
    return b_entity_impl->physicalDimension();
}

//---------------------------------------------------------------------------//
// Return the parametric dimension of the entity.
int GeometricEntity::parametricDimension() const
{ 
    return b_entity_impl->parametricDimension();
}

//---------------------------------------------------------------------------//
// Return the entity measure (volume for a 3D entity, area for 2D, and length
// for 1D). 
double GeometricEntity::measure() const
{ 
    return b_entity_impl->measure();
}

//---------------------------------------------------------------------------//
// Return the centroid of the entity.
void GeometricEntity::centroid( 
    Teuchos::ArrayView<const double>& centroid ) const
{ 
    b_entity_impl->centroid( centroid );
}

//---------------------------------------------------------------------------//
// Return the axis-aligned bounding box around the entity.
void GeometricEntity::boundingBox( Teuchos::Tuple<double,6>& bounds ) const
{ 
    b_entity_impl->boundingBox( bounds );
}

//---------------------------------------------------------------------------//
// Perform a safeguard check for mapping a point to the reference space of an
// entity using the given tolerance. 
void GeometricEntity::safeguardMapToReferenceFrame(
    const Teuchos::ParameterList& parameters,
    const Teuchos::ArrayView<const double>& point,
    MappingStatus& status ) const
{ 
    b_entity_impl->safeguardMapToReferenceFrame(
	parameters, point, status );
}

//---------------------------------------------------------------------------//
// Map a point to the reference space of an entity. Return the
// parameterized point. 
void GeometricEntity::mapToReferenceFrame( 
    const Teuchos::ParameterList& parameters,
    const Teuchos::ArrayView<const double>& point,
    const Teuchos::ArrayView<double>& reference_point,
    MappingStatus& status ) const
{ 
    b_entity_impl->mapToReferenceFrame(
	parameters, point, reference_point, status );
}

//---------------------------------------------------------------------------//
// Determine if a reference point is in the parameterized space of an entity.
bool GeometricEntity::checkPointInclusion( 
    const Teuchos::ParameterList& parameters,
    const Teuchos::ArrayView<const double>& reference_point ) const
{ 
    return b_entity_impl->checkPointInclusion( parameters, reference_point );
}

//---------------------------------------------------------------------------//
// Map a reference point to the physical space of an entity.
void GeometricEntity::mapToPhysicalFrame( 
    const Teuchos::ArrayView<const double>& reference_point,
    const Teuchos::ArrayView<double>& point ) const
{
    b_entity_impl->mapToPhysicalFrame( reference_point, point );
}

//---------------------------------------------------------------------------//
// Return a string indicating the derived object type.
std::string GeometricEntity::objectType() const
{
    return b_entity_impl->entityType();
}

//---------------------------------------------------------------------------//
// Serialize the entity into a buffer.
void GeometricEntity::serialize( 
    const Teuchos::ArrayView<char>& buffer ) const
{
    b_entity_impl->serialize( buffer );
}

//---------------------------------------------------------------------------//
// Deserialize an entity from a buffer.
void GeometricEntity::deserialize( 
    const Teuchos::ArrayView<const char>& buffer )
{
    b_entity_impl->deserialize( buffer );
}

//---------------------------------------------------------------------------//
// Check whether the underlying implementation is available.
bool GeometricEntity::isEntityImplNonnull() const
{
    return Teuchos::nonnull( b_entity_impl );
}

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//
// end DTK_GeometricEntity.cpp
//---------------------------------------------------------------------------//
